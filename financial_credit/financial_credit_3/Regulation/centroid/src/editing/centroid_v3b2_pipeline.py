"""v3b2 — v3b 보강: 밴드/안전 표본에 "다른 rule 1~2개가 안전존을 벗어난" 노이즈를 20~30% 혼합.

v3b는 8개 rule 전부 안전존(decisive)인 "깨끗한" 표본만으로 gate를 학습했다. 하지만 실데이터엔
다른 rule도 같이 탈락하는 샘플이 많다 -- gate가 "다른 rule 신호가 섞인 표현"에서도 부채비율
밴드를 읽는 법을 배우도록, 8개 rule 중 1~2개를 일부러 fail 구간에서 샘플링한 행을 섞는다.
gate 라벨은 여전히 "부채비율이 85~200인가"만 기준 (decisive 여부 무관 -- gate의 임무는
밴드 판독이지 decisive 판독이 아니므로).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


import centroid_common as cc  # noqa: E402
from src.editing.generate_locate_pairs import FEATURE_COLUMNS, AMOUNT_COLUMNS  # noqa: E402
from centroid_causal_patch import get_layer_module, build_batch, idxs_for_position  # noqa: E402
from centroid_edit_shift import GatedShiftHook  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402
from centroid_v3b_pipeline import SAFE_RANGES, AMOUNT_SAFE_HI, INFO_WIDE_MULT  # noqa: E402

RUN_DIR = cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3b2"
TARGET_LAYER = 21
NEW_THRESHOLD = 85.0
FRAC_NOISY = 0.25  # 8개 rule 중 1~2개가 안전존을 벗어나는 행의 비율

# SAFE_RANGES의 반대편(탈락) 구간 -- threshold 주변·바깥 값
FAIL_RANGES = {
    "보통주자본비율(%)": (0.0, 4.5), "기본자본비율(%)": (0.0, 6.0), "총자본비율(%)": (0.0, 8.0),
    "단순기본자본비율(%)": (0.0, 3.0), "공제후자본양수": (-AMOUNT_SAFE_HI, 0.0),
    "유동비율(%)": (10.0, 100.0), "자기자본비율(%)": (0.0, 20.0), "자본총계": (-AMOUNT_SAFE_HI, 0.0),
}


def generate_noisy_pool(rule_num: int, n_rows: int, seed: int, frac_noisy=FRAC_NOISY) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _, target_column, target_theta, *_ = cc.rule_spec(rule_num)
    other_rule_cols = list(SAFE_RANGES.keys())

    data = {}
    for col in FEATURE_COLUMNS:
        if col == target_column:
            data[col] = rng.uniform(70.0, 220.0, n_rows)
        elif col in SAFE_RANGES:
            lo, hi = SAFE_RANGES[col]
            data[col] = rng.uniform(lo, hi, n_rows)
        else:
            defaults = {"총자산순이익률ROA(%)": (0.3, 0.3), "영업흑자": (1.0, 500_000.0),
                        "이익잉여금양수": (1.0, 500_000.0), "예대율(%)": (100.0, 15.0)}
            theta, sigma = defaults[col]
            vals = rng.normal(theta, sigma * INFO_WIDE_MULT, n_rows)
            if col not in ("영업흑자", "이익잉여금양수"):
                vals = np.clip(vals, 0.0, None)
            data[col] = vals

    df = pd.DataFrame(data)

    # 노이즈 혼합: frac_noisy 비율의 행에서 무작위 1~2개 rule을 FAIL_RANGES로 덮어쓴다
    n_noisy = int(round(n_rows * frac_noisy))
    noisy_idx = rng.choice(n_rows, size=n_noisy, replace=False)
    n_violations_per_row = rng.integers(1, 3, size=n_noisy)  # 1 또는 2개
    for pos, row_i in enumerate(noisy_idx):
        k = n_violations_per_row[pos]
        chosen_cols = rng.choice(other_rule_cols, size=k, replace=False)
        for col in chosen_cols:
            lo, hi = FAIL_RANGES[col]
            df.loc[row_i, col] = rng.uniform(lo, hi)
    df["_is_noisy"] = False
    df.loc[noisy_idx, "_is_noisy"] = True
    df["_n_violated_rules"] = 0
    df.loc[noisy_idx, "_n_violated_rules"] = n_violations_per_row

    for col in AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("int64")
    percent_cols = [c for c in FEATURE_COLUMNS if c not in AMOUNT_COLUMNS]
    df[percent_cols] = df[percent_cols].round(2)
    df.insert(0, "company", [f"firm_v3b2_{i:07d}" for i in range(n_rows)])
    df.insert(0, "bank", [f"bank_v3b2_{i:07d}" for i in range(n_rows)])

    from src.dataset.check_rule_labels import compute_rule_labels
    df["label"] = compute_rule_labels(df)
    return df


def run(args):
    start_time = datetime.now()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print(f"[1/1] v3b2 노이즈 혼합 풀 생성 (n={args.n_rows}, frac_noisy={args.frac_noisy} -- "
          f"8개 rule 중 1~2개가 안전존 밖인 행 혼합, gate 라벨은 부채비율 값만 기준)")
    t0 = time.time()
    pool = generate_noisy_pool(args.rule, args.n_rows, args.seed, frac_noisy=args.frac_noisy)
    print(f"생성 완료 ({time.time()-t0:.1f}s) | noisy={pool['_is_noisy'].sum()} "
          f"clean={(~pool['_is_noisy']).sum()}")
    (RUN_DIR / "data").mkdir(parents=True, exist_ok=True)
    pool.to_csv(RUN_DIR / "data" / "v3b2_pool.csv", index=False)

    rule_column = "부채비율(%)"
    # 밴드/안전 라벨: 부채비율 값만 기준 (decisive 여부 무관 -- gate는 "밴드 판독"이 임무)
    values = pd.to_numeric(pool[rule_column], errors="coerce")
    decisive = cc.other_rules_pass_mask(pool, rule_column)  # 진단용으로만 기록
    band_value_mask = (values > NEW_THRESHOLD) & (values <= 200.0)
    safe_value_mask = values <= NEW_THRESHOLD
    pool["band"] = np.where(band_value_mask, "band", np.where(safe_value_mask, "safe", "other"))
    pool["decisive"] = decisive
    print(f"decisive={decisive.sum()} (진단용, gate 라벨엔 미반영) | {pool['band'].value_counts().to_dict()}")
    print(f"  band 중 noisy(타rule 위반 섞임)={pool[(pool['band']=='band') & pool['_is_noisy']].shape[0]} / "
          f"{pool[pool['band']=='band'].shape[0]}")
    print(f"  safe 중 noisy(타rule 위반 섞임)={pool[(pool['band']=='safe') & pool['_is_noisy']].shape[0]} / "
          f"{pool[pool['band']=='safe'].shape[0]}")

    band_df = pool[pool["band"] == "band"].reset_index(drop=True)
    safe_df = pool[pool["band"] == "safe"].reset_index(drop=True)
    if len(band_df) == 0 or len(safe_df) == 0:
        raise RuntimeError(f"밴드/안전 샘플 부족 (band={len(band_df)}, safe={len(safe_df)})")

    tokenizer, model = cc.load_model(args.model_name, device=device)

    t0 = time.time()
    cache_path = RUN_DIR / "cache_hidden.pt"
    if cache_path.exists():
        cached = torch.load(cache_path)
        x_band, x_safe = cached["x_band"], cached["x_safe"]
    else:
        n_band = min(args.n_cal_extract, len(band_df))
        n_safe = min(args.n_cal_extract, len(safe_df))
        band_sub = band_df.sample(n=n_band, random_state=args.seed).reset_index(drop=True)
        safe_sub = safe_df.sample(n=n_safe, random_state=args.seed).reset_index(drop=True)
        hid_band, _, _ = cc.extract_layer_position_hidden(model, tokenizer, band_sub, rule_column, device,
                                                            batch_size=args.batch_size)
        hid_safe, _, _ = cc.extract_layer_position_hidden(model, tokenizer, safe_sub, rule_column, device,
                                                            batch_size=args.batch_size)
        x_band = hid_band["debt_span"][:, TARGET_LAYER, :].numpy()
        x_safe = hid_safe["debt_span"][:, TARGET_LAYER, :].numpy()
        torch.save({"x_band": x_band, "x_safe": x_safe, "band_sub": band_sub, "safe_sub": safe_sub}, cache_path)
    print(f"hidden state 추출 완료 ({time.time()-t0:.1f}s) band={len(x_band)} safe={len(x_safe)}")

    X = np.concatenate([x_band, x_safe], axis=0)
    y = np.concatenate([np.ones(len(x_band)), np.zeros(len(x_safe))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료. 합성(v3b2) held-out 정확도={gate_synth_held_acc:.4f}")

    v2_shift_cfg = torch.load(cc.CENTROID_ROOT / "result" / "centroid_v2" / "edit" / "shift" / "rule1_shift_config.pt")
    delta_v = v2_shift_cfg["delta_v"]
    assert v2_shift_cfg["layer"] == TARGET_LAYER
    print(f"delta_v(v2 고정, L{TARGET_LAYER}) norm={delta_v.norm().item():.3f}")

    band_cal = band_df.sample(n=min(args.n_cal, len(band_df)), random_state=args.seed)
    safe_cal = safe_df.sample(n=min(args.n_cal, len(safe_df)), random_state=args.seed)

    def build_rows_info(sub_df):
        infos = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in sub_df.iterrows()]
        batch = build_batch(tokenizer, infos, device)
        idxs = [idxs_for_position(info, "debt_span") for info in infos]
        return batch, idxs

    batch_band, idxs_band = build_rows_info(band_cal)
    batch_safe, idxs_safe = build_rows_info(safe_cal)
    with torch.no_grad():
        clean_band_pred = model(**batch_band).logits.argmax(-1).cpu().numpy()
        clean_safe_pred = model(**batch_safe).logits.argmax(-1).cpu().numpy()

    @torch.no_grad()
    def run_gated(layer_idx, batch, idxs_per_sample, alpha):
        module = get_layer_module(model, layer_idx)
        hook = GatedShiftHook(idxs_per_sample, w_gate_t, b_gate, delta_v, alpha, ungated=False)
        handle = module.register_forward_hook(hook)
        try:
            out = model(**batch)
        finally:
            handle.remove()
        return out.logits.argmax(-1).cpu().numpy()

    cal_rows = []
    for alpha in args.alphas:
        pred_band = run_gated(TARGET_LAYER, batch_band, idxs_band, alpha)
        pred_safe = run_gated(TARGET_LAYER, batch_safe, idxs_safe, alpha)
        band_flip = float((pred_band[clean_band_pred == 1] == 0).mean()) if (clean_band_pred == 1).any() else float("nan")
        safe_false_flip = float((pred_safe[clean_safe_pred == 1] == 0).mean()) if (clean_safe_pred == 1).any() else float("nan")
        cal_rows.append({"alpha": alpha, "band_flip_rate": band_flip, "safe_false_flip_rate": safe_false_flip})
        print(f"  alpha={alpha} | 밴드 flip={band_flip:.3f} 안전 오탐={safe_false_flip:.3f}")
    cal_df = pd.DataFrame(cal_rows)
    ok = cal_df[cal_df["safe_false_flip_rate"] <= args.max_safe_false_flip]
    best_alpha = float((ok if len(ok) else cal_df).sort_values("band_flip_rate", ascending=False).iloc[0]["alpha"])
    print(f"선정된 alpha: {best_alpha}")

    print("*** test.csv를 여는 지점 (평가 전용) ***")
    df = cc.load_test_csv()
    _, rc, theta, *_ = cc.rule_spec(args.rule)
    decisive_real = cc.other_rules_pass_mask(df, rc)
    values_real = pd.to_numeric(df[rc], errors="coerce")
    band_mask_real = decisive_real & (values_real > NEW_THRESHOLD) & (values_real <= theta)
    safe_mask_real = decisive_real & (values_real <= NEW_THRESHOLD)
    cross_mask_real = ~decisive_real
    new_label = df["label"].copy()
    new_label.loc[band_mask_real] = 0

    module = get_layer_module(model, TARGET_LAYER)

    @torch.no_grad()
    def predict_full(df_full, batch_size=64):
        preds = []
        rows = df_full.reset_index(drop=True)
        for i in range(0, len(rows), batch_size):
            sub = rows.iloc[i:i + batch_size]
            infos = [cc.row_positions(tokenizer, r, rc, cc.MAX_LEN)[1] for _, r in sub.iterrows()]
            batch = build_batch(tokenizer, infos, device)
            idxs = [idxs_for_position(info, "debt_span") for info in infos]
            hook = GatedShiftHook(idxs, w_gate_t, b_gate, delta_v, best_alpha, ungated=False)
            handle = module.register_forward_hook(hook)
            try:
                out = model(**batch)
            finally:
                handle.remove()
            preds.extend(out.logits.argmax(-1).cpu().tolist())
        return np.array(preds)

    t0 = time.time()
    full_preds = predict_full(df, batch_size=args.eval_batch_size)
    print(f"전체 평가 완료 ({time.time()-t0:.1f}s)")

    from sklearn.metrics import accuracy_score, f1_score
    labels = new_label.values
    full_result = {
        "accuracy": float(accuracy_score(labels, full_preds)),
        "macro_f1": float(f1_score(labels, full_preds, average="macro")),
        "efficacy_band": float(accuracy_score(labels[band_mask_real.values], full_preds[band_mask_real.values])),
        "locality_safe": float(accuracy_score(labels[safe_mask_real.values], full_preds[safe_mask_real.values])),
        "locality_cross": float(accuracy_score(labels[cross_mask_real.values], full_preds[cross_mask_real.values])),
    }
    print(f"[v3b2 전체 평가] {full_result}")

    rng = np.random.default_rng(args.seed)
    real_band_idx = rng.choice(df.index[band_mask_real], size=min(300, band_mask_real.sum()), replace=False)
    real_safe_idx = rng.choice(df.index[safe_mask_real], size=min(300, safe_mask_real.sum()), replace=False)
    real_band_df = df.loc[real_band_idx].reset_index(drop=True)
    real_safe_df = df.loc[real_safe_idx].reset_index(drop=True)
    _, hb = cc.get_cached_hidden(RUN_DIR / "cache_real_band.pt", model, tokenizer, real_band_df, rc, device)
    _, hs = cc.get_cached_hidden(RUN_DIR / "cache_real_safe.pt", model, tokenizer, real_safe_df, rc, device)
    xb, xs = hb[:, TARGET_LAYER, :].numpy(), hs[:, TARGET_LAYER, :].numpy()
    gate_pred = np.concatenate([((xb @ w_gate + b_gate) > 0).astype(int), ((xs @ w_gate + b_gate) > 0).astype(int)])
    gate_true = np.concatenate([np.ones(len(xb)), np.zeros(len(xs))])
    gate_real_acc = float((gate_pred == gate_true).mean())
    print(f"gate 실데이터 정확도: {gate_real_acc:.4f}")

    report = {
        "method": "v3b2_noisy_mix", "rule": args.rule, "layer": TARGET_LAYER,
        "n_rows": args.n_rows, "frac_noisy": args.frac_noisy,
        "pool_band_counts": pool["band"].value_counts().to_dict(),
        "gate_synth_heldout_acc": gate_synth_held_acc, "gate_real_acc": gate_real_acc,
        "best_alpha": best_alpha, "calibration": cal_rows, "full_evaluation": full_result,
    }
    with open(RUN_DIR / "v3b2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate, "delta_v": delta_v, "layer": TARGET_LAYER,
                "best_alpha": best_alpha}, RUN_DIR / "v3b2_shift_config.pt")

    cc.write_manifest(RUN_DIR, script="centroid_v3b2_pipeline.py", inputs=vars(args),
                       extra={"report_summary": full_result}, start_time=start_time)
    print(f"저장 완료: {RUN_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--n_rows", type=int, default=12000)
    parser.add_argument("--frac_noisy", type=float, default=FRAC_NOISY)
    parser.add_argument("--n_cal_extract", type=int, default=1000)
    parser.add_argument("--n_cal", type=int, default=150)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max_safe_false_flip", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default=cc.LOCAL_MODEL_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
