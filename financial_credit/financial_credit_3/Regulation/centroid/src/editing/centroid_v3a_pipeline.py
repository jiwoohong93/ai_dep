"""v3a — 시도1: gaussian 재료를 interpolation해서 "다양한 맥락" 합성 데이터 확보.

1. generate_train_gaussian.py로 대량 생성 (검증된 sigma, threshold 중심)
2. generate_train_interpolation.py를 test.csv가 아니라 **1의 출력**에 대해 실행
   -> 합성 gaussian 행끼리 섞은 것이므로 test.csv를 전혀 건드리지 않음
3. 1(가중치 1) : 2(가중치 2) 로 풀 구성, 각 행의 실제 부채비율 값으로 band/safe 라벨 재계산
4. gate 학습 (L21 debt_span, delta_v는 v2 고정값 재사용)
5. test.csv 전체로 평가 (여기서만 test.csv 사용)

데이터 생성(1~2)은 CPU로 충분 (subprocess로 기존 스크립트 그대로 재사용).
"""
import argparse
import json
import subprocess
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
from centroid_causal_patch import get_layer_module, build_batch, idxs_for_position  # noqa: E402
from centroid_edit_shift import GatedShiftHook  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402
from src.models.test import format_prompt  # noqa: E402

RUN_DIR = cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3a"
TARGET_LAYER = 21
NEW_THRESHOLD = 85.0


def run_generation(args, data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    gaussian_path = data_dir / "gaussian.csv"
    interp_path = data_dir / "interpolated.csv"

    print(f"[1/2] generate_train_gaussian.py 실행 (n={args.n_gaussian}, seed=42, 검증된 sigma)")
    subprocess.run(
        [sys.executable, "src/dataset/generate_train_gaussian.py",
         "--output_path", str(gaussian_path), "--num_rows", str(args.n_gaussian), "--seed", "42"],
        cwd=cc.REGULATION_ROOT, check=True,
    )

    n_interp = args.n_gaussian * 2  # 1(gaussian):2(interpolated)
    print(f"[2/2] generate_train_interpolation.py 실행 -- input=gaussian.csv (test.csv 아님), n={n_interp}")
    subprocess.run(
        [sys.executable, "src/dataset/generate_train_interpolation.py",
         "--input_path", str(gaussian_path), "--output_path", str(interp_path),
         "--num_rows", str(n_interp), "--seed", "42"],
        cwd=cc.REGULATION_ROOT, check=True,
    )
    return gaussian_path, interp_path


def build_band_labeled_pool(gaussian_path, interp_path, seed=0):
    """gaussian+interpolation 풀은 나머지 12개 지표의 다양한 맥락을 제공하지만, 부채비율
    자체는 N(200,15)에 갇혀 85 근처(200에서 7.7시그마)에는 사실상 도달 못한다(band=n안전=0
    으로 실패 확인됨). 타깃 컬럼만 v3b와 동일하게 70~220 균일로 덮어써 커버리지를 보장하고
    (이러면 v3a/v3b의 차이가 순수하게 "다른 12개 지표를 어떻게 만들었나"로 좁혀진다),
    나머지 12개 지표는 gaussian+interpolation 값을 그대로 둔다."""
    gauss_df = pd.read_csv(gaussian_path)
    interp_df = pd.read_csv(interp_path)
    gauss_df["_source"] = "gaussian"
    interp_df["_source"] = "interpolated"
    pool = pd.concat([gauss_df, interp_df], ignore_index=True)

    rule_column = "부채비율(%)"
    rng = np.random.default_rng(seed)
    pool[rule_column] = rng.uniform(70.0, 220.0, len(pool)).round(2)
    from src.dataset.check_rule_labels import compute_rule_labels
    pool["label"] = compute_rule_labels(pool)

    decisive = cc.other_rules_pass_mask(pool, rule_column)
    values = pd.to_numeric(pool[rule_column], errors="coerce")
    band_mask = decisive & (values > NEW_THRESHOLD) & (values <= 200.0)
    safe_mask = decisive & (values <= NEW_THRESHOLD)
    pool["band"] = np.where(band_mask, "band", np.where(safe_mask, "safe", "other"))
    return pool


def run(args):
    start_time = datetime.now()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    t0 = time.time()
    data_dir = RUN_DIR / "data"
    gaussian_path, interp_path = run_generation(args, data_dir)
    print(f"데이터 생성 완료 ({time.time()-t0:.1f}s)")

    pool = build_band_labeled_pool(gaussian_path, interp_path, seed=args.seed)
    print(f"풀 구성: 전체 {len(pool)} | {pool['_source'].value_counts().to_dict()} | "
          f"{pool['band'].value_counts().to_dict()}")

    band_df = pool[pool["band"] == "band"].reset_index(drop=True)
    safe_df = pool[pool["band"] == "safe"].reset_index(drop=True)
    if len(band_df) == 0 or len(safe_df) == 0:
        raise RuntimeError(f"밴드/안전 샘플이 부족함 (band={len(band_df)}, safe={len(safe_df)})")

    rule_column = "부채비율(%)"
    tokenizer, model = cc.load_model(args.model_name, device=device)

    # ---------------- hidden state 추출 (L21 debt_span) ----------------
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

    # ---------------- gate 학습 ----------------
    X = np.concatenate([x_band, x_safe], axis=0)
    y = np.concatenate([np.ones(len(x_band)), np.zeros(len(x_safe))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료. 합성(v3a) held-out 정확도={gate_synth_held_acc:.4f}")

    # ---------------- delta_v: v2 고정값 재사용 ----------------
    v2_shift_cfg = torch.load(cc.CENTROID_ROOT / "result" / "centroid_v2" / "edit" / "shift" / "rule1_shift_config.pt")
    delta_v = v2_shift_cfg["delta_v"]
    assert v2_shift_cfg["layer"] == TARGET_LAYER
    print(f"delta_v(v2 고정, L{TARGET_LAYER}) norm={delta_v.norm().item():.3f}")

    # ---------------- alpha calibration (합성만) ----------------
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

    # ---------------- test.csv 전체 평가 (여기서만 사용) ----------------
    print("*** test.csv를 여는 지점 (평가 전용) ***")
    df = cc.load_test_csv()
    _, rc, theta, *_ = cc.rule_spec(args.rule)
    decisive = cc.other_rules_pass_mask(df, rc)
    values = pd.to_numeric(df[rc], errors="coerce")
    band_mask_real = decisive & (values > NEW_THRESHOLD) & (values <= theta)
    safe_mask_real = decisive & (values <= NEW_THRESHOLD)
    cross_mask_real = ~decisive
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
    print(f"[v3a 전체 평가] {full_result}")

    # ---------------- gate 실데이터 정확도 (진단) ----------------
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
        "method": "v3a_gaussian_interpolation", "rule": args.rule, "layer": TARGET_LAYER,
        "n_gaussian": args.n_gaussian, "n_interpolated": args.n_gaussian * 2,
        "pool_source_counts": pool["_source"].value_counts().to_dict(),
        "pool_band_counts": pool["band"].value_counts().to_dict(),
        "gate_synth_heldout_acc": gate_synth_held_acc, "gate_real_acc": gate_real_acc,
        "best_alpha": best_alpha, "calibration": cal_rows, "full_evaluation": full_result,
    }
    with open(RUN_DIR / "v3a_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate, "delta_v": delta_v, "layer": TARGET_LAYER,
                "best_alpha": best_alpha}, RUN_DIR / "v3a_shift_config.pt")

    cc.write_manifest(RUN_DIR, script="centroid_v3a_pipeline.py", inputs=vars(args),
                       extra={"report_summary": full_result}, start_time=start_time)
    print(f"저장 완료: {RUN_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--n_gaussian", type=int, default=4000)
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
