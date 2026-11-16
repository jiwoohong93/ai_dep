"""relationship(기혼 프록시) 중립화 프로브 — "기혼 신호가 사라진 세계"의 DEO/acc.

test 전 행의 relationship/marital-status를 동일 값으로 통일해 예측한다.
변수가 상수가 되면 판별력이 사라지므로, "모델이 이 변수를 못 쓰게 하는 편집"의
효과를 근사한다. 두 방향으로 괄호를 잡는다:

- unmarried: 전원 (Not-in-family, Never-married) — 기혼 보너스 제거 방향
- married  : 전원 (Husband/Wife 성별 정합, Married-civ-spouse) — 전원 기혼 취급 방향

출력(--output_dir): metrics.json (원본 대비 acc/DEO/TPR 변화), predictions.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import format_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",
                   default=str(BIAS_ROOT / "outputs/finetuned_xlm-roberta-large_cgmix_rel/best"))
    p.add_argument("--test_path", default=str(BIAS_ROOT / "data/test/text/test.csv"))
    p.add_argument("--neutral_mode", choices=["unmarried", "married"], required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size, device):
    preds = []
    for s in tqdm(range(0, len(texts), batch_size), desc="Predict", leave=False):
        enc = tokenizer(texts[s : s + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def group_stats(df, preds):
    y = df["label"].to_numpy()
    male = (df["sex"] == "Male").to_numpy()
    tpr_m = float((preds[male & (y == 1)] == 1).mean())
    tpr_f = float((preds[~male & (y == 1)] == 1).mean())
    return {
        "accuracy": float((preds == y).mean()),
        "tpr_male": tpr_m, "tpr_female": tpr_f, "deo_lit": abs(tpr_m - tpr_f),
        "acc_male": float((preds[male] == y[male]).mean()),
        "acc_female": float((preds[~male] == y[~male]).mean()),
    }


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(BIAS_ROOT / "result/llm_cgmix_rel_training"
                              / f"neutral_relationship_{args.neutral_mode}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.test_path)

    neut = df.copy()
    if args.neutral_mode == "unmarried":
        neut["relationship"] = "Not-in-family"
        neut["marital-status"] = "Never-married"
    else:
        neut["relationship"] = np.where(neut["sex"] == "Male", "Husband", "Wife")
        neut["marital-status"] = "Married-civ-spouse"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    pred_o = predict(model, tokenizer, df.apply(format_prompt, axis=1).tolist(),
                     args.batch_size, device)
    pred_n = predict(model, tokenizer, neut.apply(format_prompt, axis=1).tolist(),
                     args.batch_size, device)

    metrics = {
        "model": args.model_name, "neutral_mode": args.neutral_mode,
        "original": group_stats(df, pred_o),
        "neutralized": group_stats(df, pred_n),
        "pred_change_rate": float((pred_o != pred_n).mean()),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"index": df.index, "pred": pred_o, "pred_neutral": pred_n}).to_csv(
        out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
