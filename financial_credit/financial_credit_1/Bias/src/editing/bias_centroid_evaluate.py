"""Exp5-Bias: 최종 평가 -- Bias/test.csv 16,281건 전체(평가 전용, 여기서 처음 사용).

README 성공 기준 확인:
  - 편집 후 DEO_lit < 0.05
  - 예측 정확도 >= 0.82, 편집으로 인한 하락 <= 1%p
  - cg rule 무손상 (cg_boundary_fixed/random.csv로 확인)
  - 비혼 recall 남녀 공통 +3.3%p 근접 여부
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


import bias_centroid_common as bc  # noqa: E402
from bias_centroid_hooks import get_layer_module, build_batch_infos, build_batch  # noqa: E402
from bias_centroid_edit import GatedShiftBiasHook  # noqa: E402

OUT_DIR = bc.RESULT_ROOT / "evaluate"


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
            idxs_m = [info["marital-status"] for info in infos]
            idxs_r = [info["relationship"] for info in infos]
            hook = hook_factory(idxs_m, idxs_r)
            handle = module.register_forward_hook(hook)
        try:
            out = model(**batch)
        finally:
            if handle is not None:
                handle.remove()
        preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def recall_by_marital(df, preds, unmarried_mask):
    y = df["label"].to_numpy()
    male = (df["sex"] == "Male").to_numpy()
    out = {}
    for name, mask in [("unmarried", unmarried_mask), ("married", ~unmarried_mask)]:
        for sex_name, sex_mask in [("all", np.ones(len(df), dtype=bool)), ("male", male), ("female", ~male)]:
            m = mask & sex_mask & (y == 1)
            if m.sum() > 0:
                out[f"recall_{name}_{sex_name}"] = float((preds[m] == 1).mean())
                out[f"n_{name}_{sex_name}"] = int(m.sum())
    return out


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    cfg = torch.load(args.shift_config)
    layer = cfg["layer"]
    print(f"*** Bias/test.csv 로드 (평가 전용, 처음 사용) *** layer={layer} alpha={args.alpha}")

    df = bc.load_test_csv()
    unmarried_mask = df["marital-status"].isin(["Never-married", "Divorced", "Separated", "Widowed"]).to_numpy()

    tokenizer, model = bc.load_model(device=device)
    module = get_layer_module(model, layer)

    def hook_factory(idxs_m, idxs_r):
        return GatedShiftBiasHook(idxs_m, idxs_r, cfg["w_gate"], cfg["b_gate"],
                                   cfg["delta_v_marital"], cfg["delta_v_relationship"], args.alpha)

    t0 = time.time()
    pred_base = predict_full(model, tokenizer, df, device, batch_size=args.batch_size)
    print(f"base 예측 완료 ({time.time()-t0:.1f}s)")
    t0 = time.time()
    pred_edit = predict_full(model, tokenizer, df, device, module=module, hook_factory=hook_factory,
                              batch_size=args.batch_size)
    print(f"edit 예측 완료 ({time.time()-t0:.1f}s)")

    labels = df["label"].to_numpy()

    def summarize(preds, tag):
        m = {
            "tag": tag, "n": int(len(df)),
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            **bc.fairness_metrics(df, preds),
            **recall_by_marital(df, preds, unmarried_mask),
        }
        return m

    result_base = summarize(pred_base, "base")
    result_edit = summarize(pred_edit, "edited")
    print(f"[base ] acc={result_base['accuracy']:.4f} deo_lit={result_base.get('deo_lit'):.4f} "
          f"recall_unmarried_all={result_base.get('recall_unmarried_all'):.4f}")
    print(f"[edit ] acc={result_edit['accuracy']:.4f} deo_lit={result_edit.get('deo_lit'):.4f} "
          f"recall_unmarried_all={result_edit.get('recall_unmarried_all'):.4f}")
    print(f"acc 변화: {result_edit['accuracy']-result_base['accuracy']:+.4f} "
          f"(기준: <=1%p 하락 허용)")
    print(f"비혼 recall 변화(전체): "
          f"{result_edit.get('recall_unmarried_all',0)-result_base.get('recall_unmarried_all',0):+.4f} "
          f"(목표: +0.033 근처)")

    # ---------------- cg rule 무손상 확인 ----------------
    cg_results = {}
    for name in ("cg_boundary_fixed", "cg_boundary_random"):
        path = bc.BIAS_ROOT / "data" / "test" / "text" / f"{name}.csv"
        if not path.exists():
            continue
        cgdf = pd.read_csv(path)
        cg_pred_base = predict_full(model, tokenizer, cgdf, device, batch_size=args.batch_size)
        cg_pred_edit = predict_full(model, tokenizer, cgdf, device, module=module,
                                     hook_factory=hook_factory, batch_size=args.batch_size)
        cg_labels = cgdf["label"].to_numpy()
        cg_results[name] = {
            "n": int(len(cgdf)),
            "acc_base": float(accuracy_score(cg_labels, cg_pred_base)),
            "acc_edit": float(accuracy_score(cg_labels, cg_pred_edit)),
            "pred_changed_rate": float((cg_pred_base != cg_pred_edit).mean()),
        }
        print(f"[cg rule 확인: {name}] acc_base={cg_results[name]['acc_base']:.4f} "
              f"acc_edit={cg_results[name]['acc_edit']:.4f} "
              f"pred 변화율={cg_results[name]['pred_changed_rate']:.4f} (기준: 거의 0)")

    report = {"config": str(args.shift_config), "layer": layer, "alpha": args.alpha,
              "base": result_base, "edited": result_edit, "cg_rule_check": cg_results}
    with open(OUT_DIR / f"report_layer{layer}_alpha{args.alpha}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUT_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift_config",
                         default=str(bc.RESULT_ROOT / "edit" / "layer9_seed0" / "shift_config.pt"))
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
