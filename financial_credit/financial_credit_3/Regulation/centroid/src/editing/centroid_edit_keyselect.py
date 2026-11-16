"""Exp 4.0 (EDRM centroid): key 선택성 사전 검증 — 편집 전 필수.

L21(hidden_states 인덱스) debt_span 경로의 rank-1 weight edit(Exp4b)은 FFN
intermediate(gelu 후, 4096차원, output.dense의 입력)를 key 공간으로 쓴다. 세 후보 key:
  k1 = mu_band            (밴드(85~200) 생성 샘플 centroid)
  k2 = mu_band - mu_safe  (차이 벡터)
  k3 = band-vs-safe linear probe의 raw-space weight 벡터

각 key에 대해 kᵀC⁻¹h (C: 안전(<=85, 생성) + 타rule 탈락(실데이터) covariance, "keep set")
분포를 4그룹(밴드/안전/타rule탈락/실데이터밴드)에서 그려 AUC를 계산하고, 최고 key를
Exp4b에 넘긴다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


import centroid_common as cc  # noqa: E402

ENCODER_LAYER_IDX = 20  # hidden_states L21 == encoder.layer[20] 출력


def probe_weight_raw_space(X, y):
    """StandardScaler + LogisticRegression을 raw 공간 (w_raw, b_raw)로 환산."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xs, y)
    w_std = clf.coef_[0]
    b_std = clf.intercept_[0]
    w_raw = w_std / scaler.scale_
    b_raw = b_std - float(np.dot(w_std, scaler.mean_ / scaler.scale_))
    return w_raw, b_raw


def run(args):
    cc.ensure_dirs()
    out_dir = cc.RESULT_ROOT / "edit" / "keyselect"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, rule_column, theta, *_ = cc.rule_spec(args.rule)
    device = f"cuda:{args.gpu_id}"

    synth_path = cc.RESULT_ROOT / "gen_samples" / f"rule{args.rule}_synthetic_samples.csv"
    synth = pd.read_csv(synth_path)
    df = cc.load_test_csv()

    tokenizer, model = cc.load_model(args.model_name, device=device)

    # ---------------- FFN intermediate 추출 (4개 그룹) ----------------
    band_synth = synth[synth["band"] == "band"].reset_index(drop=True)
    safe_synth = synth[synth["band"] == "safe"].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    other_fail_mask = cc.other_rule_fail_mask(df, args.rule)
    other_fail_idx = rng.choice(df.index[other_fail_mask], size=min(args.n_real, other_fail_mask.sum()),
                                 replace=False)
    other_fail_real = df.loc[other_fail_idx].reset_index(drop=True)

    band_real_mask = cc.real_band_mask(df, args.rule, args.new_threshold)
    band_real_idx = rng.choice(df.index[band_real_mask], size=min(args.n_real, band_real_mask.sum()),
                                replace=False)
    band_real = df.loc[band_real_idx].reset_index(drop=True)

    print(f"그룹 크기: 생성-밴드={len(band_synth)} 생성-안전={len(safe_synth)} "
          f"실데이터-타rule탈락={len(other_fail_real)} 실데이터-밴드={len(band_real)}")

    def cached_ffn(name, sub_df):
        path = cc.CACHE_DIR / f"ffn_L{ENCODER_LAYER_IDX+1}_rule{args.rule}_{name}.pt"
        return cc.get_cached_ffn_intermediate(path, model, tokenizer, sub_df, rule_column,
                                               ENCODER_LAYER_IDX, device, batch_size=args.batch_size)

    x_band_synth = cached_ffn("band_synth", band_synth).numpy()
    x_safe_synth = cached_ffn("safe_synth", safe_synth).numpy()
    x_other_fail = cached_ffn("other_fail_real", other_fail_real).numpy()
    x_band_real = cached_ffn("band_real", band_real).numpy()

    # ---------------- key 3종 ----------------
    mu_band = x_band_synth.mean(0)
    mu_safe = x_safe_synth.mean(0)
    k1 = mu_band
    k2 = mu_band - mu_safe
    X_probe = np.concatenate([x_band_synth, x_safe_synth], axis=0)
    y_probe = np.concatenate([np.ones(len(x_band_synth)), np.zeros(len(x_safe_synth))])
    w_raw, b_raw = probe_weight_raw_space(X_probe, y_probe)
    k3 = w_raw
    keys = {"k1_mu_band": k1, "k2_mu_band_minus_safe": k2, "k3_probe_weight": k3}

    # ---------------- keep set covariance (안전 + 타rule 탈락) ----------------
    keep = np.concatenate([x_safe_synth, x_other_fail], axis=0)
    keep_c = keep - keep.mean(0, keepdims=True)
    C = (keep_c.T @ keep_c) / len(keep)
    damping = args.damping_rel * np.trace(C) / C.shape[0]
    C_inv = np.linalg.inv(C + damping * np.eye(C.shape[0]))
    print(f"keep set n={len(keep)}  damping={damping:.4f}")

    groups = {
        "생성-밴드": x_band_synth, "생성-안전": x_safe_synth,
        "실데이터-타rule탈락": x_other_fail, "실데이터-밴드": x_band_real,
    }

    report = {"rule": args.rule, "encoder_layer_idx": ENCODER_LAYER_IDX, "keys": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (kname, kvec) in zip(axes, keys.items()):
        w = C_inv @ kvec  # (4096,)
        scores = {g: (x @ w) for g, x in groups.items()}

        auc1 = roc_auc_score(
            np.concatenate([np.ones(len(scores["생성-밴드"])), np.zeros(len(scores["생성-안전"]))]),
            np.concatenate([scores["생성-밴드"], scores["생성-안전"]]))
        auc2 = roc_auc_score(
            np.concatenate([np.ones(len(scores["생성-밴드"])), np.zeros(len(scores["실데이터-타rule탈락"]))]),
            np.concatenate([scores["생성-밴드"], scores["실데이터-타rule탈락"]]))
        auc3 = roc_auc_score(
            np.concatenate([np.ones(len(scores["실데이터-밴드"])), np.zeros(len(scores["생성-안전"]))]),
            np.concatenate([scores["실데이터-밴드"], scores["생성-안전"]]))
        auc4 = roc_auc_score(
            np.concatenate([np.ones(len(scores["실데이터-밴드"])), np.zeros(len(scores["실데이터-타rule탈락"]))]),
            np.concatenate([scores["실데이터-밴드"], scores["실데이터-타rule탈락"]]))
        avg_auc = float(np.mean([auc1, auc2, auc3, auc4]))
        min_auc = float(np.min([auc1, auc2, auc3, auc4]))

        report["keys"][kname] = {
            "auc_synthband_vs_synthsafe": float(auc1),
            "auc_synthband_vs_realotherfail": float(auc2),
            "auc_realband_vs_synthsafe": float(auc3),
            "auc_realband_vs_realotherfail": float(auc4),
            "avg_auc": avg_auc, "min_auc": min_auc,
        }
        print(f"[{kname}] auc: synth밴드vs안전={auc1:.4f} synth밴드vs타실패={auc2:.4f} "
              f"실밴드vs안전={auc3:.4f} 실밴드vs타실패={auc4:.4f} | avg={avg_auc:.4f} min={min_auc:.4f}")

        for g, s in scores.items():
            ax.hist(s, bins=25, alpha=0.55, label=f"{g} (n={len(s)})", density=True)
        ax.set_title(f"{kname}\navg AUC={avg_auc:.3f}")
        ax.legend(fontsize=7)
        ax.set_xlabel("k^T C^-1 h")
    axes[0].set_ylabel("density")
    fig.suptitle(f"rule{args.rule} L{ENCODER_LAYER_IDX+1} FFN-intermediate key 발화량 분포 (keep-set C 정규화)")
    fig.tight_layout()
    fig.savefig(out_dir / f"rule{args.rule}_key_firing_distributions.png", dpi=150)
    plt.close(fig)

    best_key = max(report["keys"], key=lambda k: report["keys"][k]["min_auc"])
    report["selected_key"] = best_key
    print(f"\n선정된 key (min_auc 최고): {best_key}")

    with open(out_dir / f"rule{args.rule}_keyselect_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({
        "keys": {k: torch.tensor(v, dtype=torch.float32) for k, v in keys.items()},
        "selected_key": best_key,
        "C": torch.tensor(C, dtype=torch.float32),
        "C_inv": torch.tensor(C_inv, dtype=torch.float32),
        "encoder_layer_idx": ENCODER_LAYER_IDX,
    }, out_dir / f"rule{args.rule}_keys.pt")

    print(f"저장 완료: {out_dir}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--new_threshold", type=float, default=85.0)
    parser.add_argument("--n_real", type=int, default=300)
    parser.add_argument("--damping_rel", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default=cc.LOCAL_MODEL_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
