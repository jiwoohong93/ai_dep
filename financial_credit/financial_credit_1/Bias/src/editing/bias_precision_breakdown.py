"""1-1: 보정 전후 비혼/기혼 x 남/여 4그룹 precision(+recall) 분해표.

alpha 후보 3개(0.85, 0.9, 1.0)를 모두 평가한다 -- 1-2 감사 결과 alpha=1.0은 test.csv
결과를 직접 보고 고른 것(절차 오염)이라, 순수 합성 calibration만으로 보간한 alpha=0.9
(README 목표 +3.3%p 역산)와 나란히 비교해야 어느 쪽이 실제로 방어 가능한 선택인지 판단할
수 있다. 0.85는 참고용 하한.

확인 포인트: 보정 후 비혼 precision이 기혼 precision에 수렴하는지(=캘리브레이션 교정
완결의 증거) 아니면 역전(과승인)하는지(=alpha를 낮춰야 한다는 근거)이다.

test.csv는 여기서도 평가 전용으로만 사용 -- 이 스크립트가 새로 학습/선택하는 것은 없음
(기존 result/bias_centroid/edit/layer9_seed0/shift_config.pt를 그대로 로드).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


import bias_centroid_common as bc  # noqa: E402
from bias_centroid_hooks import get_layer_module  # noqa: E402
from bias_centroid_edit import GatedShiftBiasHook  # noqa: E402
from bias_centroid_evaluate import predict_full  # noqa: E402


def precision(preds, labels, mask):
    m = mask & (preds == 1)
    if m.sum() == 0:
        return float("nan"), 0
    return float((labels[m] == 1).mean()), int(m.sum())


def recall(preds, labels, mask):
    m = mask & (labels == 1)
    if m.sum() == 0:
        return float("nan"), 0
    return float((preds[m] == 1).mean()), int(m.sum())


def run(args):
    start_time = datetime.now()
    run_dir = bc.new_run_dir(bc.RESULT_ROOT / "precision_breakdown", "run")
    device = f"cuda:{args.gpu_id}"

    cfg = torch.load(args.shift_config)
    layer = cfg["layer"]
    print(f"config={args.shift_config} layer={layer} alphas={args.alphas}")

    df = bc.load_test_csv()
    unmarried_mask = df["marital-status"].isin(["Never-married", "Divorced", "Separated", "Widowed"]).to_numpy()
    male_mask = (df["sex"] == "Male").to_numpy()
    labels = df["label"].to_numpy()
    groups = {
        "비혼-남성": unmarried_mask & male_mask,
        "비혼-여성": unmarried_mask & ~male_mask,
        "기혼-남성": ~unmarried_mask & male_mask,
        "기혼-여성": ~unmarried_mask & ~male_mask,
    }

    tokenizer, model = bc.load_model(device=device)
    module = get_layer_module(model, layer)

    pred_base = predict_full(model, tokenizer, df, device, batch_size=args.batch_size)

    all_rows = []
    base_stats = {}
    for name, mask in groups.items():
        p, n_p = precision(pred_base, labels, mask)
        r, n_r = recall(pred_base, labels, mask)
        base_stats[name] = {"precision": p, "recall": r}
        print(f"[base] {name:10s} precision={p:.4f}(n_pred1={n_p}) recall={r:.4f}(n_qualified={n_r})")

    for alpha in args.alphas:
        def hook_factory(idxs_m, idxs_r, a=alpha):
            return GatedShiftBiasHook(idxs_m, idxs_r, cfg["w_gate"], cfg["b_gate"],
                                       cfg["delta_v_marital"], cfg["delta_v_relationship"], a)
        pred_edit = predict_full(model, tokenizer, df, device, module=module, hook_factory=hook_factory,
                                  batch_size=args.batch_size)
        print(f"\n=== alpha={alpha} ===")
        for name, mask in groups.items():
            p, n_p = precision(pred_edit, labels, mask)
            r, n_r = recall(pred_edit, labels, mask)
            db = base_stats[name]
            row = {"alpha": alpha, "group": name,
                   "precision_base": db["precision"], "precision_edit": p, "precision_delta": p - db["precision"],
                   "recall_base": db["recall"], "recall_edit": r, "recall_delta": r - db["recall"]}
            all_rows.append(row)
            print(f"  {name:10s} precision {db['precision']:.4f} -> {p:.4f} ({p-db['precision']:+.4f}) | "
                  f"recall {db['recall']:.4f} -> {r:.4f} ({r-db['recall']:+.4f})")

        gap_before = base_stats["비혼-남성"]["precision"] - base_stats["기혼-남성"]["precision"]
        gap_after_m = next(r["precision_edit"] for r in all_rows if r["alpha"] == alpha and r["group"] == "비혼-남성") - \
                      next(r["precision_edit"] for r in all_rows if r["alpha"] == alpha and r["group"] == "기혼-남성")
        gap_before_f = base_stats["비혼-여성"]["precision"] - base_stats["기혼-여성"]["precision"]
        gap_after_f = next(r["precision_edit"] for r in all_rows if r["alpha"] == alpha and r["group"] == "비혼-여성") - \
                      next(r["precision_edit"] for r in all_rows if r["alpha"] == alpha and r["group"] == "기혼-여성")
        verdict_m = "수렴" if abs(gap_after_m) < abs(gap_before) else ("역전" if gap_after_m < 0 else "격차 유지/확대")
        verdict_f = "수렴" if abs(gap_after_f) < abs(gap_before_f) else ("역전" if gap_after_f < 0 else "격차 유지/확대")
        print(f"  [남성] precision 격차(비혼-기혼): {gap_before:+.4f} -> {gap_after_m:+.4f}  [{verdict_m}]")
        print(f"  [여성] precision 격차(비혼-기혼): {gap_before_f:+.4f} -> {gap_after_f:+.4f}  [{verdict_f}]")

    import pandas as pd
    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(run_dir / "precision_recall_breakdown.csv", index=False)
    with open(run_dir / "precision_recall_breakdown.json", "w", encoding="utf-8") as f:
        json.dump({"base": base_stats, "by_alpha": all_rows}, f, ensure_ascii=False, indent=2)

    bc.write_manifest(run_dir, script="bias_precision_breakdown.py", inputs=vars(args),
                       extra={"layer": layer}, start_time=start_time)
    print(f"\n저장 완료: {run_dir}")
    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift_config",
                         default=str(bc.RESULT_ROOT / "edit" / "layer9_seed0" / "shift_config.pt"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.85, 0.9, 1.0])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
