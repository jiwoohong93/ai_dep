"""EDRM centroid v2 — Exp4(a): gated shift hook, 순수 합성 데이터로 delta_v/gate 계산.

delta_v: locate pairs(base=통과/source=탈락, 500개 subsample)의 L21 centroid 차이.
gate: centroid_v2_gen_samples.py가 만든 합성 band/safe로 학습, held-out도 합성으로.
실데이터는 evaluate 단계(Exp5)에서만 진단용으로 잠깐 확인 (튜닝에는 전혀 안 씀).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import centroid_common as cc  # noqa: E402
import centroid_v2_common as v2  # noqa: E402
from centroid_causal_patch import get_layer_module, build_batch, idxs_for_position  # noqa: E402
from centroid_edit_shift import GatedShiftHook  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402

TARGET_LAYER = 21


def build_rows_info(tokenizer, sub_df, rule_column, device):
    infos = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in sub_df.iterrows()]
    batch = build_batch(tokenizer, infos, device)
    idxs = [idxs_for_position(info, "debt_span") for info in infos]
    return infos, batch, idxs


@torch.no_grad()
def run_gated(model, layer_idx, batch, idxs_per_sample, w_gate, b_gate, delta_v, alpha, ungated=False):
    module = get_layer_module(model, layer_idx)
    hook = GatedShiftHook(idxs_per_sample, w_gate, b_gate, delta_v, alpha, ungated)
    handle = module.register_forward_hook(hook)
    try:
        out = model(**batch)
    finally:
        handle.remove()
    preds = out.logits.argmax(-1).cpu().numpy()
    return preds, np.array(hook.gate_fired)


def run(args):
    v2.ensure_dirs_v2()
    out_dir = v2.RESULT_ROOT_V2 / "edit" / "shift"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, rule_column, theta, *_ = cc.rule_spec(args.rule)
    device = f"cuda:{args.gpu_id}"
    tokenizer, model = cc.load_model(args.model_name, device=device)

    # ---------------- delta_v: locate pairs(합성) 500개 subsample, L21 ----------------
    base_df_full, source_df_full, _ = v2.load_locate_pairs(args.rule)
    select_rng = np.random.default_rng(0)
    select_idx = select_rng.choice(len(base_df_full), size=500, replace=False)
    cache_base = v2.CACHE_DIR_V2 / f"locate_pairs_rule{args.rule}_base.pt"
    cache_source = v2.CACHE_DIR_V2 / f"locate_pairs_rule{args.rule}_source.pt"
    full_base = torch.load(cache_base)
    full_source = torch.load(cache_source)
    delta_v = full_source["debt_span"][:, TARGET_LAYER, :].mean(0) - full_base["debt_span"][:, TARGET_LAYER, :].mean(0)
    print(f"delta_v(합성 locate pairs, L{TARGET_LAYER}) norm={delta_v.norm().item():.3f}")

    # ---------------- gate: 합성 band/safe(Exp3 v2 산출물)로 학습 ----------------
    synth = pd.read_csv(v2.RESULT_ROOT_V2 / "gen_samples" / f"rule{args.rule}_synthetic_samples.csv")
    span_cache = torch.load(v2.CACHE_DIR_V2 / f"gen_samples_rule{args.rule}_hidden.pt")["debt_span"]
    x_all = span_cache[:, TARGET_LAYER, :].numpy()
    band_mask = (synth["band"] == "band").values
    safe_mask = (synth["band"] == "safe").values

    X = np.concatenate([x_all[band_mask], x_all[safe_mask]], axis=0)
    y = np.concatenate([np.ones(band_mask.sum()), np.zeros(safe_mask.sum())])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료(전부 합성). 합성 held-out 정확도={gate_synth_held_acc:.4f}")

    # ---------------- alpha 스윕 (calibration=합성만) ----------------
    band_df = synth[synth["band"] == "band"].reset_index(drop=True)
    safe_df = synth[synth["band"] == "safe"].reset_index(drop=True)
    cal_band = band_df.sample(n=min(args.n_cal, len(band_df)), random_state=args.seed)
    cal_safe = safe_df.sample(n=min(args.n_cal, len(safe_df)), random_state=args.seed)

    _, batch_band, idxs_band = build_rows_info(tokenizer, cal_band, rule_column, device)
    _, batch_safe, idxs_safe = build_rows_info(tokenizer, cal_safe, rule_column, device)
    with torch.no_grad():
        clean_band_pred = model(**batch_band).logits.argmax(-1).cpu().numpy()
        clean_safe_pred = model(**batch_safe).logits.argmax(-1).cpu().numpy()

    rows = []
    for alpha in args.alphas:
        for gated in (True, False):
            pred_band, fired_band = run_gated(model, TARGET_LAYER, batch_band, idxs_band,
                                               w_gate_t, b_gate, delta_v, alpha, ungated=not gated)
            pred_safe, fired_safe = run_gated(model, TARGET_LAYER, batch_safe, idxs_safe,
                                               w_gate_t, b_gate, delta_v, alpha, ungated=not gated)
            band_flip = float((pred_band[clean_band_pred == 1] == 0).mean()) if (clean_band_pred == 1).any() else float("nan")
            safe_false_flip = float((pred_safe[clean_safe_pred == 1] == 0).mean()) if (clean_safe_pred == 1).any() else float("nan")
            rows.append({"alpha": alpha, "gated": gated, "band_flip_rate": band_flip,
                         "safe_false_flip_rate": safe_false_flip,
                         "gate_fire_rate_band": float(fired_band.mean()), "gate_fire_rate_safe": float(fired_safe.mean())})
            print(f"  alpha={alpha} gated={gated} | 밴드 flip={band_flip:.3f} 안전 오탐={safe_false_flip:.3f} "
                  f"| gate발동(밴드/안전)={fired_band.mean():.2f}/{fired_safe.mean():.2f}")

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(out_dir / f"rule{args.rule}_alpha_calibration.csv", index=False)
    gated_rows = cal_df[cal_df["gated"]]
    ok = gated_rows[gated_rows["safe_false_flip_rate"] <= args.max_safe_false_flip]
    best_alpha = float((ok if len(ok) else gated_rows).sort_values("band_flip_rate", ascending=False).iloc[0]["alpha"])
    print(f"\n선정된 alpha: {best_alpha}")

    report = {
        "rule": args.rule, "layer": TARGET_LAYER, "data_source": "100% synthetic",
        "gate_synth_heldout_acc": gate_synth_held_acc,
        "delta_v_norm": float(delta_v.norm().item()), "best_alpha": best_alpha, "calibration": rows,
    }
    with open(out_dir / f"rule{args.rule}_shift_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate, "delta_v": delta_v, "layer": TARGET_LAYER,
                "best_alpha": best_alpha}, out_dir / f"rule{args.rule}_shift_config.pt")
    print(f"저장 완료: {out_dir}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--n_cal", type=int, default=150)
    parser.add_argument("--max_safe_false_flip", type=float, default=0.02)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default=cc.LOCAL_MODEL_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
