"""v4 gate 개선 (GPU0/1로 배정됐던 부분, 실제로는 CPU로 충분) —
threshold 스윕 / 앙상블(OR·AND·확률평균) / 통합 재학습(v3a+v3b2 데이터 합본).

전부 centroid_v4_precompute.py가 만든 pred_base/pred_shift_a1과, 이미 있는 test.csv L21
공유 캐시(result/centroid/shared_cache)만으로 계산된다 -- 새 forward pass가 전혀 필요 없다:
샘플 i의 최종 예측 = gate_fires(i) ? pred_shift_a1[i] : pred_base[i].
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
from sklearn.metrics import accuracy_score, f1_score


import centroid_common as cc  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402

TARGET_LAYER = 21
PRECOMPUTE_PATH = cc.CENTROID_ROOT / "result" / "centroid" / "v4_precompute" / "rule1_precomputed.pt"
SHARED_CACHE = cc.CENTROID_ROOT / "result" / "centroid" / "shared_cache" / "test_full_rule1_hidden.pt"
OUT_DIR = cc.CENTROID_ROOT / "result" / "centroid" / "v4_free_variants"


def combine(fire_mask, pred_base, pred_shift):
    return np.where(fire_mask, pred_shift, pred_base)


def metrics(preds, labels, band_mask, safe_mask, cross_mask):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "efficacy_band": float(accuracy_score(labels[band_mask], preds[band_mask])),
        "locality_safe": float(accuracy_score(labels[safe_mask], preds[safe_mask])),
        "locality_cross": float(accuracy_score(labels[cross_mask], preds[cross_mask])),
    }


def wait_and_load():
    if not PRECOMPUTE_PATH.exists():
        raise FileNotFoundError(f"{PRECOMPUTE_PATH} 없음 -- centroid_v4_precompute.py(GPU3) 먼저 실행")
    pre = torch.load(PRECOMPUTE_PATH)
    cache = torch.load(SHARED_CACHE)
    return pre, cache


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pre, cache = wait_and_load()
    pred_base, pred_shift_a1 = pre["pred_base"], pre["pred_shift_a1"]
    labels = pre["labels"]
    band_mask, safe_mask, cross_mask = pre["band_mask"], pre["safe_mask"], pre["cross_mask"]
    x_full = cache["debt_span"][:, TARGET_LAYER, :].numpy()  # (38160, 1024)
    print(f"사전계산 로드 완료. n={len(pred_base)}")

    gates = {}
    for name, path in [("v3a", cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3a" / "v3a_shift_config.pt"),
                        ("v3b2", cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3b2" / "v3b2_shift_config.pt")]:
        cfg = torch.load(path)
        gates[name] = {"w": cfg["w_gate"].numpy(), "b": float(cfg["b_gate"])}

    scores = {name: x_full @ g["w"] + g["b"] for name, g in gates.items()}  # raw logit score

    results = []

    # ---------------- baseline (threshold=0, 기존 gate) ----------------
    for name in ("v3a", "v3b2"):
        fire = scores[name] > 0
        preds = combine(fire, pred_base, pred_shift_a1)
        m = metrics(preds, labels, band_mask, safe_mask, cross_mask)
        m.update({"method": f"{name}_threshold0(baseline)"})
        results.append(m)
        print(f"[{name} baseline] {m}")

    # ---------------- 1. threshold 스윕 ----------------
    sweep_rows = {"v3a": [], "v3b2": []}
    for name in ("v3a", "v3b2"):
        for tau in args.thresholds:
            shift_logit = np.log(tau / (1 - tau))  # sigmoid(score) > tau <=> score > logit(tau)
            fire = scores[name] > shift_logit
            preds = combine(fire, pred_base, pred_shift_a1)
            m = metrics(preds, labels, band_mask, safe_mask, cross_mask)
            m["tau"] = tau
            sweep_rows[name].append(m)
            print(f"[{name} tau={tau}] acc={m['accuracy']:.4f} efficacy={m['efficacy_band']:.4f} "
                  f"locality={m['locality_safe']:.4f}")
        pd.DataFrame(sweep_rows[name]).to_csv(OUT_DIR / f"rule1_threshold_sweep_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in [("v3a", "#2f7d76"), ("v3b2", "#c0392b")]:
        df_s = pd.DataFrame(sweep_rows[name])
        ax.plot(df_s["efficacy_band"], df_s["locality_safe"], marker="o", label=name, color=color)
        for _, row in df_s.iterrows():
            ax.annotate(f"{row['tau']:.1f}", (row["efficacy_band"], row["locality_safe"]), fontsize=7)
    ax.set_xlabel("efficacy(밴드)"); ax.set_ylabel("locality(안전)")
    ax.set_title("threshold 스윕: efficacy-locality 곡선 (라벨=tau)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rule1_threshold_sweep_curve.png", dpi=150)
    plt.close(fig)

    best_thr = {}
    for name in ("v3a", "v3b2"):
        df_s = pd.DataFrame(sweep_rows[name])
        best = df_s.sort_values("accuracy", ascending=False).iloc[0]
        best_thr[name] = best.to_dict()
        r = dict(best.to_dict()); r["method"] = f"{name}_threshold_best(tau={best['tau']})"
        results.append(r)
        print(f"[{name} 최적 threshold] tau={best['tau']} acc={best['accuracy']:.4f}")

    # ---------------- 2. 앙상블 ----------------
    fire_a = scores["v3a"] > 0
    fire_b = scores["v3b2"] > 0
    for ens_name, fire in [("ensemble_OR", fire_a | fire_b), ("ensemble_AND", fire_a & fire_b)]:
        preds = combine(fire, pred_base, pred_shift_a1)
        m = metrics(preds, labels, band_mask, safe_mask, cross_mask)
        m["method"] = ens_name
        results.append(m)
        print(f"[{ens_name}] {m}")

    prob_avg = (1 / (1 + np.exp(-scores["v3a"])) + 1 / (1 + np.exp(-scores["v3b2"]))) / 2
    fire_prob = prob_avg > 0.5
    preds = combine(fire_prob, pred_base, pred_shift_a1)
    m = metrics(preds, labels, band_mask, safe_mask, cross_mask)
    m["method"] = "ensemble_prob_avg"
    results.append(m)
    print(f"[ensemble_prob_avg] {m}")

    # ---------------- 3. 통합 재학습 (v3a+v3b2 캐시된 표본 합본) ----------------
    v3a_cache = torch.load(cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3a" / "cache_hidden.pt")
    v3b2_cache = torch.load(cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3b2" / "cache_hidden.pt")
    x_band_combined = np.concatenate([v3a_cache["x_band"], v3b2_cache["x_band"]], axis=0)
    x_safe_combined = np.concatenate([v3a_cache["x_safe"], v3b2_cache["x_safe"]], axis=0)
    X = np.concatenate([x_band_combined, x_safe_combined], axis=0)
    y = np.concatenate([np.ones(len(x_band_combined)), np.zeros(len(x_safe_combined))])
    w_unified, b_unified = probe_weight_raw_space(X, y)
    print(f"통합 gate 학습 완료 (band={len(x_band_combined)} safe={len(x_safe_combined)})")

    score_unified = x_full @ w_unified + b_unified
    fire_unified = score_unified > 0
    preds = combine(fire_unified, pred_base, pred_shift_a1)
    m = metrics(preds, labels, band_mask, safe_mask, cross_mask)
    m["method"] = "unified_gate(v3a+v3b2 합본)"
    results.append(m)
    print(f"[unified_gate] {m}")

    unified_sweep = []
    for tau in args.thresholds:
        shift_logit = np.log(tau / (1 - tau))
        fire = score_unified > shift_logit
        preds = combine(fire, pred_base, pred_shift_a1)
        mm = metrics(preds, labels, band_mask, safe_mask, cross_mask)
        mm["tau"] = tau
        unified_sweep.append(mm)
    pd.DataFrame(unified_sweep).to_csv(OUT_DIR / "rule1_threshold_sweep_unified.csv", index=False)
    best_unified = pd.DataFrame(unified_sweep).sort_values("accuracy", ascending=False).iloc[0]
    r = dict(best_unified.to_dict()); r["method"] = f"unified_gate_threshold_best(tau={best_unified['tau']})"
    results.append(r)
    print(f"[unified_gate 최적 threshold] tau={best_unified['tau']} acc={best_unified['accuracy']:.4f}")

    torch.save({"w_gate": torch.tensor(w_unified, dtype=torch.float32), "b_gate": float(b_unified),
                "delta_v": pre["delta_v"], "layer": TARGET_LAYER, "best_alpha": 1.0,
                "best_tau": float(best_unified["tau"])}, OUT_DIR / "rule1_unified_gate_config.pt")

    with open(OUT_DIR / "rule1_all_variants.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    pd.DataFrame(results).to_csv(OUT_DIR / "rule1_all_variants.csv", index=False)

    print(f"\n=== 전체 acc 기준 정렬 ===")
    df_res = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print(df_res[["method", "accuracy", "efficacy_band", "locality_safe"]].to_string(index=False))
    print(f"저장 완료: {OUT_DIR}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+",
                         default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
