"""cg 편집 alpha 스윕 + 이중 평가 프레임.

(a) rule 기준: 전체 test.csv에 "capital-gain >= NEW_THETA(7700)"라는 순수 rule label을
    새로 매겨서, 편집된 모델의 예측이 이 rule을 얼마나 잘 재현하는지 측정 (정책이 의도한
    대로 반영됐는가 -- 실제 소득과 무관하게 "규칙 자체를 얼마나 잘 따르는가").
(b) 실제 소득 기준: 기존처럼 test.csv의 진짜 label(연소득 실측치) 대비 정확도 (현실 정합 비용).

band(=[7000,7700), 옛rule 발동/새rule 비발동) / above(>=7700, 둘 다 발동) / below(<7000, 둘 다
비발동) 세 구간으로 나눠서 (a)(b) 둘 다 구간별로 보고한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score


import bias_centroid_common as bc  # noqa: E402
from bias_centroid_hooks import get_layer_module, build_batch_infos, build_batch  # noqa: E402
from bias_centroid_edit_cg import GatedShiftCgHook, OLD_THETA, NEW_THETA  # noqa: E402

OUT_DIR = bc.RESULT_ROOT / "cg_alpha_sweep"


@torch.no_grad()
def predict_full(model, tokenizer, df, device, module=None, hook_factory=None, batch_size=64):
    preds = []
    rows = df.reset_index(drop=True)
    for i in range(0, len(rows), batch_size):
        sub = rows.iloc[i:i + batch_size]
        texts, infos = build_batch_infos(tokenizer, sub)
        batch = build_batch(tokenizer, infos, device)
        handle = None
        if hook_factory is not None:
            idxs = [info["capital-gain"] for info in infos]
            handle = module.register_forward_hook(hook_factory(idxs))
        try:
            out = model(**batch)
        finally:
            if handle is not None:
                handle.remove()
        preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    cfg = torch.load(args.cg_config)
    layer = cfg["layer"]
    print(f"layer={layer} alphas={args.alphas}")

    df = bc.load_test_csv()
    cg = df["capital-gain"].to_numpy()
    band_mask = (cg >= OLD_THETA) & (cg < NEW_THETA)
    above_mask = cg >= NEW_THETA
    below_mask = cg < OLD_THETA
    real_label = df["label"].to_numpy()
    rule_label = (cg >= NEW_THETA).astype(int)  # (a) 순수 rule 기준 라벨

    print(f"band={band_mask.sum()} above={above_mask.sum()} below={below_mask.sum()}")

    tokenizer, model = bc.load_model(device=device)
    module = get_layer_module(model, layer)

    pred_base = predict_full(model, tokenizer, df, device, batch_size=args.batch_size)

    def region_metrics(preds, mask, ref_label):
        if mask.sum() == 0:
            return {"n": 0, "accuracy": float("nan")}
        return {"n": int(mask.sum()), "accuracy": float(accuracy_score(ref_label[mask], preds[mask]))}

    rows = []
    # alpha=0 (편집 전) 기준선도 같이 기록
    for alpha in [0.0] + list(args.alphas):
        if alpha == 0.0:
            pred = pred_base
        else:
            def hook_factory(idxs, a=alpha):
                return GatedShiftCgHook(idxs, cfg["w_gate"], cfg["b_gate"], cfg["delta_v"], a)
            pred = predict_full(model, tokenizer, df, device, module=module, hook_factory=hook_factory,
                                 batch_size=args.batch_size)

        base_band_pos = pred_base[band_mask] == 1
        n_base_band_pos = max(int(base_band_pos.sum()), 1)
        efficacy_band = float((base_band_pos & (pred[band_mask] == 0)).sum() / n_base_band_pos)
        locality_below = float((pred[below_mask] == pred_base[below_mask]).mean())
        locality_above = float((pred[above_mask] == pred_base[above_mask]).mean())

        row = {
            "alpha": alpha,
            "efficacy_band(1->0 flip)": efficacy_band,
            "locality_below(예측 불변율)": locality_below,
            "locality_above(예측 불변율)": locality_above,
            # (a) rule 기준
            "acc_vs_RULE_all": float(accuracy_score(rule_label, pred)),
            "acc_vs_RULE_band": region_metrics(pred, band_mask, rule_label)["accuracy"],
            "acc_vs_RULE_above": region_metrics(pred, above_mask, rule_label)["accuracy"],
            "acc_vs_RULE_below": region_metrics(pred, below_mask, rule_label)["accuracy"],
            # (b) 실제 소득 기준
            "acc_vs_REAL_all": float(accuracy_score(real_label, pred)),
            "acc_vs_REAL_band": region_metrics(pred, band_mask, real_label)["accuracy"],
            "acc_vs_REAL_above": region_metrics(pred, above_mask, real_label)["accuracy"],
            "acc_vs_REAL_below": region_metrics(pred, below_mask, real_label)["accuracy"],
        }
        rows.append(row)
        print(f"alpha={alpha:.2f} | efficacy_band={efficacy_band:.4f} "
              f"loc_below={locality_below:.4f} loc_above={locality_above:.4f} | "
              f"acc_RULE(전체)={row['acc_vs_RULE_all']:.4f} acc_RULE(band)={row['acc_vs_RULE_band']:.4f} | "
              f"acc_REAL(전체)={row['acc_vs_REAL_all']:.4f} acc_REAL(band)={row['acc_vs_REAL_band']:.4f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "alpha_sweep.csv", index=False)
    with open(OUT_DIR / "alpha_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUT_DIR}")
    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg_config",
                         default=str(bc.RESULT_ROOT / "edit_cg/layer15_seed0/shift_config.pt"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7, 1.0, 1.3, 1.5, 1.75, 2.0])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
