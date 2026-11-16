"""Exp1-Bias: capital-gain(rule) 신호와 비혼(marital) 신호를 layer x position으로 동시 스캔.

README 확정 목표: "cg rule은 반응 안 하면서 비혼 표지가 반응하는 위치"를 찾아야 하므로,
Regulation Exp1(단일 신호 스캔)과 달리 이번엔 두 신호를 같은 (layer, position)에서 동시에
측정해 분리(marital 신호는 크고 cg 신호는 작은 지점)를 직접 찾는다.

marital/cg 쌍 모두 train.csv에서만 생성(test.csv 무사용, bias_gen_pairs.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


import bias_centroid_common as bc  # noqa: E402
import bias_gen_pairs as bp  # noqa: E402

POSITIONS = ["cls", "marital-status", "relationship", "capital-gain"]


def standardized_distance(xa, xb):
    mu_a, mu_b = xa.mean(0), xb.mean(0)
    var_a, var_b = xa.var(0), xb.var(0)
    pooled_std = np.clip(np.sqrt((var_a + var_b) / 2.0), 1e-6, None)
    return float(np.linalg.norm((mu_a - mu_b) / pooled_std))


def l2_distance(xa, xb):
    return float(np.linalg.norm(xa.mean(0) - xb.mean(0)))


def probe_accuracy(xa, xb, seed=0):
    X = np.concatenate([xa, xb], axis=0)
    y = np.concatenate([np.ones(len(xa)), np.zeros(len(xb))])
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def run(args):
    out_dir = bc.RESULT_ROOT / "locate_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    print(f"[seed={args.seed}] device={device} n_pairs={args.n_pairs}")

    train = bc.load_train_csv()
    married, unmarried, _ = bp.sample_marital_pairs(train, n=args.n_pairs, seed=args.seed)
    cg_above, cg_below, _ = bp.sample_cg_pairs(train, n=args.n_pairs, seed=args.seed + 1000)

    tokenizer, model = bc.load_model(device=device)

    t0 = time.time()
    h_married, _, _ = bc.extract_multi_position_hidden(model, tokenizer, married, device,
                                                         batch_size=args.batch_size, positions=POSITIONS)
    h_unmarried, _, _ = bc.extract_multi_position_hidden(model, tokenizer, unmarried, device,
                                                           batch_size=args.batch_size, positions=POSITIONS)
    print(f"marital 쌍 hidden state 추출 완료 ({time.time()-t0:.1f}s)")

    t0 = time.time()
    h_above, _, _ = bc.extract_multi_position_hidden(model, tokenizer, cg_above, device,
                                                       batch_size=args.batch_size, positions=POSITIONS)
    h_below, _, _ = bc.extract_multi_position_hidden(model, tokenizer, cg_below, device,
                                                       batch_size=args.batch_size, positions=POSITIONS)
    print(f"cg 쌍 hidden state 추출 완료 ({time.time()-t0:.1f}s)")

    rows = []
    for pos in POSITIONS:
        for layer in range(bc.N_HIDDEN_STATES):
            xa_m = h_married[pos][:, layer, :].numpy()
            xb_m = h_unmarried[pos][:, layer, :].numpy()
            xa_c = h_above[pos][:, layer, :].numpy()
            xb_c = h_below[pos][:, layer, :].numpy()

            marital_std = standardized_distance(xa_m, xb_m)
            marital_acc = probe_accuracy(xa_m, xb_m, seed=args.seed)
            cg_std = standardized_distance(xa_c, xb_c)
            cg_acc = probe_accuracy(xa_c, xb_c, seed=args.seed)

            rows.append({
                "seed": args.seed, "layer": layer, "position": pos,
                "marital_std_dist": marital_std, "marital_probe_acc": marital_acc,
                "cg_std_dist": cg_std, "cg_probe_acc": cg_acc,
                "separation": marital_std - cg_std,
                "separation_ratio": marital_std / max(cg_std, 1e-6),
            })
            print(f"  layer={layer:2d} pos={pos:15s} marital_std={marital_std:7.3f} "
                  f"(acc={marital_acc:.3f}) cg_std={cg_std:7.3f} (acc={cg_acc:.3f}) "
                  f"sep={marital_std-cg_std:7.3f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / f"scan_seed{args.seed}.csv", index=False)

    top = result_df.sort_values("separation", ascending=False).head(10)
    print("\n=== 분리(marital_std - cg_std) 상위 10 ===")
    print(top[["layer", "position", "marital_std_dist", "cg_std_dist", "separation"]].to_string(index=False))
    top.to_json(out_dir / f"top10_seed{args.seed}.json", orient="records", indent=2)

    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(4 * len(POSITIONS), 6), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        sub = result_df[result_df.position == pos].sort_values("layer")
        ax.plot(sub.marital_std_dist, sub.layer, label="marital", color="#c0392b")
        ax.plot(sub.cg_std_dist, sub.layer, label="cg rule", color="#2f7169")
        ax.set_title(pos)
        ax.set_xlabel("standardized distance")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    axes[0].set_ylabel("layer")
    fig.suptitle(f"Exp1-Bias: marital vs cg 신호 (seed={args.seed})")
    fig.tight_layout()
    fig.savefig(out_dir / f"scan_curves_seed{args.seed}.png", dpi=140)

    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
