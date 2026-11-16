"""Exp5-cg: rule(cg 7,000->7,700) 편집 최종 평가 + bias 편집과의 2x2 상호작용 확인.

test.csv는 실제 소득 라벨이라 "cg rule 자체의 정답"이 없다 (Regulation과 달리 rule이 곧
ground truth가 아님) -- 그래서 efficacy/locality는 실제 라벨이 아니라 "예측이 편집 의도대로
움직였는가"(행동 지표)로 잰다:
  efficacy_band  = band(7000<=cg<7700) 표본 중 1->0으로 flip된 비율 (정책 변경이 실제로 적용)
  locality_nonband = band 밖(cg<7000 or cg>=7700) 표본 중 예측이 안 바뀐 비율
전체 정확도(실제 소득 라벨 기준)도 같이 보고해 부수 피해를 확인한다.

2x2: {cg 편집 off/on} x {bias 편집 off/on} 네 조합을 전부 평가해 두 편집이 서로의 효과를
훼손하지 않는지(DEO_lit, band efficacy 둘 다) 직접 확인한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


import bias_centroid_common as bc  # noqa: E402
from bias_centroid_hooks import get_layer_module, build_batch_infos, build_batch  # noqa: E402
from bias_centroid_edit import GatedShiftBiasHook  # noqa: E402
from bias_centroid_edit_cg import GatedShiftCgHook, OLD_THETA, NEW_THETA  # noqa: E402

OUT_DIR = bc.RESULT_ROOT / "evaluate_cg"


@torch.no_grad()
def predict_full(model, tokenizer, df, device, hooks=(), batch_size=64):
    """hooks: [(module, hook_factory), ...] -- hook_factory(infos)->hook 객체."""
    preds = []
    rows = df.reset_index(drop=True)
    for i in range(0, len(rows), batch_size):
        sub = rows.iloc[i:i + batch_size]
        texts, infos = build_batch_infos(tokenizer, sub)
        batch = build_batch(tokenizer, infos, device)
        handles = []
        for module, hook_factory in hooks:
            handles.append(module.register_forward_hook(hook_factory(infos)))
        try:
            out = model(**batch)
        finally:
            for h in handles:
                h.remove()
        preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    cfg_bias = torch.load(args.bias_config)
    cfg_cg = torch.load(args.cg_config)
    layer_bias, layer_cg = cfg_bias["layer"], cfg_cg["layer"]
    print(f"bias: layer={layer_bias} alpha={args.alpha_bias} | cg: layer={layer_cg} alpha={args.alpha_cg}")

    df = bc.load_test_csv()
    cg_vals = df["capital-gain"].to_numpy()
    band_mask = (cg_vals >= OLD_THETA) & (cg_vals < NEW_THETA)
    nonband_mask = ~band_mask
    labels = df["label"].to_numpy()

    tokenizer, model = bc.load_model(device=device)
    module_bias = get_layer_module(model, layer_bias)
    module_cg = get_layer_module(model, layer_cg)

    def bias_hook_factory(infos):
        idxs_m = [info["marital-status"] for info in infos]
        idxs_r = [info["relationship"] for info in infos]
        return GatedShiftBiasHook(idxs_m, idxs_r, cfg_bias["w_gate"], cfg_bias["b_gate"],
                                   cfg_bias["delta_v_marital"], cfg_bias["delta_v_relationship"], args.alpha_bias)

    def cg_hook_factory(infos):
        idxs = [info["capital-gain"] for info in infos]
        return GatedShiftCgHook(idxs, cfg_cg["w_gate"], cfg_cg["b_gate"], cfg_cg["delta_v"], args.alpha_cg)

    combos = {
        "off_off": [],
        "cg_on": [(module_cg, cg_hook_factory)],
        "bias_on": [(module_bias, bias_hook_factory)],
        "both_on": [(module_cg, cg_hook_factory), (module_bias, bias_hook_factory)],
    }

    preds = {}
    for name, hooks in combos.items():
        preds[name] = predict_full(model, tokenizer, df, device, hooks=hooks, batch_size=args.batch_size)
        print(f"[{name}] 예측 완료")

    unmarried_mask = df["marital-status"].isin(["Never-married", "Divorced", "Separated", "Widowed"]).to_numpy()

    def metrics(p, tag):
        base_band_pos = preds["off_off"][band_mask] == 1
        n_base_band_pos = max(int(base_band_pos.sum()), 1)
        flipped = (base_band_pos) & (p[band_mask] == 0)
        m = {
            "tag": tag,
            "accuracy": float(accuracy_score(labels, p)),
            "efficacy_band(cg 1->0 flip)": float(flipped.sum() / n_base_band_pos),
            "locality_nonband(cg 예측 불변율)": float((p[nonband_mask] == preds["off_off"][nonband_mask]).mean()),
            **bc.fairness_metrics(df, p),
            "recall_unmarried_all": float((p[unmarried_mask & (labels == 1)] == 1).mean()),
        }
        return m

    results = {name: metrics(p, name) for name, p in preds.items()}
    for name, m in results.items():
        print(f"[{name:8s}] acc={m['accuracy']:.4f} efficacy_band={m['efficacy_band(cg 1->0 flip)']:.4f} "
              f"locality_nonband={m['locality_nonband(cg 예측 불변율)']:.4f} "
              f"deo_lit={m.get('deo_lit', float('nan')):.4f} recall_unmarried={m['recall_unmarried_all']:.4f}")

    # ---------------- 상호간섭 판정 ----------------
    interference = {
        "cg_efficacy_손상_by_bias": results["both_on"]["efficacy_band(cg 1->0 flip)"] - results["cg_on"]["efficacy_band(cg 1->0 flip)"],
        "bias_deo_손상_by_cg": results["both_on"]["deo_lit"] - results["bias_on"]["deo_lit"],
        "bias_recall_손상_by_cg": results["both_on"]["recall_unmarried_all"] - results["bias_on"]["recall_unmarried_all"],
    }
    print("\n=== 상호간섭 (0에 가까울수록 서로 무간섭) ===")
    for k, v in interference.items():
        print(f"  {k}: {v:+.4f}")

    report = {"bias_config": str(args.bias_config), "cg_config": str(args.cg_config),
              "alpha_bias": args.alpha_bias, "alpha_cg": args.alpha_cg,
              "results_2x2": results, "interference": interference}
    with open(OUT_DIR / f"report_2x2_abias{args.alpha_bias}_acg{args.alpha_cg}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장 완료: {OUT_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias_config",
                         default=str(bc.RESULT_ROOT / "edit/layer9_seed0/shift_config.pt"))
    parser.add_argument("--cg_config",
                         default=str(bc.RESULT_ROOT / "edit_cg/layer15_seed0/shift_config.pt"))
    parser.add_argument("--alpha_bias", type=float, default=0.8)
    parser.add_argument("--alpha_cg", type=float, default=1.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
