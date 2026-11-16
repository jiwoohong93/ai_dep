"""capital-gain >= 7000 soft rule을 모델이 학습했는지 진단하는 프로브.

두 가지 측정:

1) 합성 스윕: test에서 (capital-gain=0, label=0)인 실제 문맥 N개를 뽑아
   capital-gain 값만 그리드로 갈아끼운 프롬프트를 만들고 예측을 본다.
   - 학습에 나온 스파이크 값(5178, 6849, 7298, 7688, 15024)과
     학습에 없는 갭 값(6900, 7100 등)을 모두 포함 → 암기 vs 임계 일반화 구분
   - 기대: rule을 배웠다면 cg >= 7000대 값에서 문맥과 무관하게 pred=1
2) 실데이터 확인: test에서 capital-gain >= 7000인 실제 행(656행)의 정확도
   (라벨이 99% 1이므로 사실상 "rule 발동률")

출력(--output_dir): sweep_curve.csv, sweep_curve.png, metrics.json
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

# 학습 스파이크 값(*)과 미관측 갭 값을 섞은 그리드
GRID = [0, 1000, 2000, 3000, 4000, 5000, 5178, 6000, 6500, 6849, 6900, 7000,
        7100, 7298, 7500, 7688, 8000, 9000, 10000, 12000, 15024, 20000,
        30000, 50000, 99999]
SEEN = {5178, 6849, 7298, 7688, 15024, 99999}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",
                   default=str(BIAS_ROOT / "outputs/finetuned_xlm-roberta-large/best"))
    p.add_argument("--test_path", default=str(BIAS_ROOT / "data/test/text/test.csv"))
    p.add_argument("--output_dir", default=str(BIAS_ROOT / "result/analysis/cg_rule_probe"))
    p.add_argument("--n_contexts", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size, device):
    preds = []
    for s in tqdm(range(0, len(texts), batch_size), desc="Predict", leave=False):
        enc = tokenizer(texts[s : s + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    # 1) 합성 스윕 -----------------------------------------------------------
    pool = df[(df["capital-gain"] == 0) & (df["label"] == 0)]
    contexts = pool.sample(args.n_contexts, random_state=args.seed).reset_index(drop=True)
    rows, meta = [], []
    for value in GRID:
        ctx = contexts.copy()
        ctx["capital-gain"] = value
        rows.extend(ctx.apply(format_prompt, axis=1).tolist())
        meta.extend([(i, value) for i in range(len(ctx))])
    preds = predict(model, tokenizer, rows, args.batch_size, device)
    sweep = pd.DataFrame(meta, columns=["context", "cg"])
    sweep["pred"] = preds

    curve = sweep.groupby("cg")["pred"].mean().rename("p_pred1").reset_index()
    curve["seen_in_train"] = curve["cg"].isin(SEEN)
    curve.to_csv(out_dir / "sweep_curve.csv", index=False)

    # 문맥별 유효 임계값: 이 값 이상에서 계속 1로 유지되기 시작하는 최소 그리드 값
    flips = []
    piv = sweep.pivot(index="context", columns="cg", values="pred")[GRID]
    for _, r in piv.iterrows():
        arr = r.to_numpy()
        thr = None
        for k in range(len(GRID)):
            if arr[k:].all():
                thr = GRID[k]
                break
        flips.append(thr)
    flips = pd.Series(flips, dtype="float")

    # 2) 실데이터 rule 발동률 -------------------------------------------------
    hi = df[df["capital-gain"] >= 7000]
    hi_preds = predict(model, tokenizer, hi.apply(format_prompt, axis=1).tolist(),
                       args.batch_size, device)
    real_high = {
        "n": int(len(hi)),
        "acc_vs_label": float((hi_preds == hi["label"].to_numpy()).mean()),
        "pred1_rate": float((hi_preds == 1).mean()),
    }

    metrics = {
        "model": args.model_name,
        "n_contexts": args.n_contexts,
        "sweep_pred1_at_or_above_7100": float(sweep[sweep.cg >= 7100]["pred"].mean()),
        "sweep_pred1_unseen_high": float(
            sweep[(sweep.cg >= 7100) & (~sweep.cg.isin(SEEN))]["pred"].mean()),
        "sweep_pred1_seen_high": float(
            sweep[(sweep.cg >= 7100) & (sweep.cg.isin(SEEN))]["pred"].mean()),
        "sweep_pred1_below_5000": float(sweep[sweep.cg <= 5000]["pred"].mean()),
        "effective_threshold_median": (None if flips.isna().all()
                                       else float(flips.median())),
        "contexts_never_flip": int(flips.isna().sum()),
        "real_test_cg_ge_7000": real_high,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        seen = curve[curve.seen_in_train]
        unseen = curve[~curve.seen_in_train]
        ax.plot(curve.cg, curve.p_pred1, color="#bbb", lw=1, zorder=1)
        ax.scatter(unseen.cg, unseen.p_pred1, color="#4878d0", label="unseen value", zorder=2)
        ax.scatter(seen.cg, seen.p_pred1, color="#d65f5f", marker="s",
                   label="train spike value", zorder=2)
        ax.axvline(7000, color="k", ls="--", lw=1)
        ax.set_xscale("symlog", linthresh=10000)
        ax.set_xlabel("capital-gain (symlog)")
        ax.set_ylabel("P(pred=1) over contexts")
        ax.set_title("cg sweep on label-0 contexts — does crossing 7000 flip the verdict?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "sweep_curve.png", dpi=130)
    except Exception as e:  # 그림 실패해도 지표는 남김
        print(f"figure skipped: {e}")

    print(json.dumps(metrics, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
