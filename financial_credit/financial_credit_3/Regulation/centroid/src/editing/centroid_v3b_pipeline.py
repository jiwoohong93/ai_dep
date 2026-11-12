"""v3b — 시도2: 13개 지표를 전부 넓게(gaussian 확장 + uniform 혼합) 흔드는 새 생성기.

test.csv/실템플릿 전혀 미사용. 각 지표를 threshold 중심 gaussian(검증된 sigma의 2.5배)과
넓은 uniform을 50/50으로 섞어 매 행 독립적으로 샘플링 (좁은 밀집 방지). 타깃 rule(부채비율)만
70~220 균일 분포로 강제 커버.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import centroid_common as cc  # noqa: E402
from src.editing.generate_locate_pairs import FEATURE_COLUMNS, RULE_SPECS, AMOUNT_COLUMNS  # noqa: E402
from centroid_causal_patch import get_layer_module, build_batch, idxs_for_position  # noqa: E402
from centroid_edit_shift import GatedShiftHook  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402

RUN_DIR = cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3b"
TARGET_LAYER = 21
NEW_THRESHOLD = 85.0
AMOUNT_SAFE_HI = 5_000_000  # 8개 rule 중 금액 지표(자본총계/공제후자본양수)의 안전존 상한
INFO_WIDE_MULT = 2.5        # 판정에 안 쓰이는 4개 정보성 지표는 여전히 넓게 흔듦

# 8개(target 제외) rule의 안전존 상/하한 -- 이 범위 안에서 균일 샘플링하면 그 rule은
# 항상 통과이므로 "decisive"가 자동 보장된다 (첫 시도는 대칭 gaussian+uniform 혼합이라
# 안전존을 벗어날 확률이 있어 decisive 수율이 12,000건 중 37건으로 붕괴했음 -- 재설계).
SAFE_RANGES = {
    "보통주자본비율(%)": (4.5, 30.0), "기본자본비율(%)": (6.0, 30.0), "총자본비율(%)": (8.0, 30.0),
    "단순기본자본비율(%)": (3.0, 15.0), "공제후자본양수": (1.0, AMOUNT_SAFE_HI),
    "유동비율(%)": (100.0, 500.0), "자기자본비율(%)": (20.0, 100.0), "자본총계": (1.0, AMOUNT_SAFE_HI),
}


def generate_wide_pool(rule_num: int, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _, target_column, target_theta, *_ = cc.rule_spec(rule_num)

    data = {}
    for col in FEATURE_COLUMNS:
        if col == target_column:
            # 타깃 rule: 70~220 균일 분포로 전 구간 강제 커버
            data[col] = rng.uniform(70.0, 220.0, n_rows)
        elif col in SAFE_RANGES:
            # 나머지 8개 rule: 안전존 "전체"에서 균일 샘플링 -> decisive 100% 보장 +
            # theta 근처에만 몰리지 않고 안전존 구석구석 다양하게 (v1의 "θ-고정+파트너 1개"보다
            # 훨씬 넓은 맥락 다양성)
            lo, hi = SAFE_RANGES[col]
            data[col] = rng.uniform(lo, hi, n_rows)
        else:
            # 판정에 안 쓰이는 4개 정보성 지표 (ROA/영업흑자/이익잉여금양수/예대율): 넓게 흔들되
            # decisive 여부에 영향 없으므로 안전존 제약 불필요
            defaults = {"총자산순이익률ROA(%)": (0.3, 0.3), "영업흑자": (1.0, 500_000.0),
                        "이익잉여금양수": (1.0, 500_000.0), "예대율(%)": (100.0, 15.0)}
            theta, sigma = defaults[col]
            vals = rng.normal(theta, sigma * INFO_WIDE_MULT, n_rows)
            if col not in ("영업흑자", "이익잉여금양수"):
                vals = np.clip(vals, 0.0, None)
            data[col] = vals

    df = pd.DataFrame(data)
    for col in AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("int64")
    percent_cols = [c for c in FEATURE_COLUMNS if c not in AMOUNT_COLUMNS]
    df[percent_cols] = df[percent_cols].round(2)
    df.insert(0, "company", [f"firm_v3b_{i:07d}" for i in range(n_rows)])
    df.insert(0, "bank", [f"bank_v3b_{i:07d}" for i in range(n_rows)])

    from src.dataset.check_rule_labels import compute_rule_labels
    df["label"] = compute_rule_labels(df)
    return df


def run(args):
    start_time = datetime.now()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print(f"[1/1] v3b 넓은 합성 풀 생성 (n={args.n_rows}, 8개 rule은 안전존 균일, "
          f"정보성 4개는 INFO_WIDE_MULT={INFO_WIDE_MULT}, 타깃 rule 70~220 균일 강제)")
    t0 = time.time()
    pool = generate_wide_pool(args.rule, args.n_rows, args.seed)
    print(f"생성 완료 ({time.time()-t0:.1f}s)")
    (RUN_DIR / "data").mkdir(parents=True, exist_ok=True)
    pool.to_csv(RUN_DIR / "data" / "v3b_pool.csv", index=False)

    rule_column = "부채비율(%)"
    decisive = cc.other_rules_pass_mask(pool, rule_column)
    values = pd.to_numeric(pool[rule_column], errors="coerce")
    band_mask = decisive & (values > NEW_THRESHOLD) & (values <= 200.0)
    safe_mask = decisive & (values <= NEW_THRESHOLD)
    pool["band"] = np.where(band_mask, "band", np.where(safe_mask, "safe", "other"))
    print(f"decisive={decisive.sum()} | {pool['band'].value_counts().to_dict()}")

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
    print(f"gate 학습 완료. 합성(v3b) held-out 정확도={gate_synth_held_acc:.4f}")

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
    print(f"[v3b 전체 평가] {full_result}")

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
        "method": "v3b_wide_generator", "rule": args.rule, "layer": TARGET_LAYER,
        "n_rows": args.n_rows, "info_wide_mult": INFO_WIDE_MULT, "safe_ranges": SAFE_RANGES,
        "pool_band_counts": pool["band"].value_counts().to_dict(),
        "gate_synth_heldout_acc": gate_synth_held_acc, "gate_real_acc": gate_real_acc,
        "best_alpha": best_alpha, "calibration": cal_rows, "full_evaluation": full_result,
    }
    with open(RUN_DIR / "v3b_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate, "delta_v": delta_v, "layer": TARGET_LAYER,
                "best_alpha": best_alpha}, RUN_DIR / "v3b_shift_config.pt")

    cc.write_manifest(RUN_DIR, script="centroid_v3b_pipeline.py", inputs=vars(args),
                       extra={"report_summary": full_result}, start_time=start_time)
    print(f"저장 완료: {RUN_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--n_rows", type=int, default=12000)
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
