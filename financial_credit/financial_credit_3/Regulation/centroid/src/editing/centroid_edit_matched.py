"""버전 일치 edit — locate와 같은 데이터 레시피로 delta_v/gate/alpha까지 계산해 편집.

원칙 (2026-07-18 확정): locate에 v3a 레시피를 썼으면 edit(delta_v·gate·보정)도 v3a
레시피를 쓴다. 기존 v3 파이프라인은 layer(21)와 delta_v를 v1/v2에서 물려받아 gate만
버전별이었는데, 이 스크립트는 버전당 전 구성요소를 같은 레시피로 맞춘다:

  1) layer 선정: result/centroid_matched/locate_synth_{version}/rule{r}_locate_report.json의
     debt_span 후보 중 min(flip_p2f, flip_f2p)가 최대인 층 (동률이면 중앙값 층)
  2) delta_v: 같은 locate 실행의 캐시 hidden(cache_A/B.pt)에서 해당 층의
     μ(탈락) − μ(통과) — v2 L21 delta_v를 재사용하지 않음
  3) gate: 해당 버전의 gate 데이터(v3a: gate_v3a/data 파일 재사용, v3b/v3b2: 동일
     생성기+동일 라벨 규칙)를 해당 층에서 학습
  4) alpha 보정: 합성만 (안전 오탐 <= max_safe_false_flip 내 밴드 flip 최대)
  5) 평가: test.csv 전체 (여기서만 실데이터)

실행: CUDA_VISIBLE_DEVICES=3 python centroid_edit_matched.py --version v3a
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import centroid_common as cc  # noqa: E402
from centroid_causal_patch import get_layer_module, build_batch, idxs_for_position  # noqa: E402
from centroid_edit_shift import GatedShiftHook  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402

NEW_THRESHOLD = 85.0
VERSIONS = ("v3a", "v3b", "v3b2")


def select_layer(report: dict):
    """debt_span 후보 중 min(양방향 flip) 최대 → 동률이면 층 번호 중앙값(상위쪽)."""
    cands = [c for c in report["candidates"] if c["position"] == "debt_span"]
    if not cands:
        raise RuntimeError("debt_span 후보 없음 — locate 결과 확인 필요")
    score = lambda c: min(c["flip_p2f_centroid"], c["flip_f2p_centroid"])
    best = max(score(c) for c in cands)
    tied = sorted(c["layer"] for c in cands if abs(score(c) - best) < 1e-9)
    return int(tied[len(tied) // 2]), cands, best


def build_gate_pool(version: str, rule_num: int, seed: int):
    """해당 버전의 gate 학습 데이터 + band/safe 라벨 (원 파이프라인과 동일 규칙)."""
    rule_column = cc.rule_spec(rule_num)[1]
    if version == "v3a":
        from centroid_v3a_pipeline import build_band_labeled_pool
        data_dir = cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3a" / "data"
        pool = build_band_labeled_pool(data_dir / "gaussian.csv", data_dir / "interpolated.csv",
                                        seed=seed)
    elif version == "v3b":
        from centroid_v3b_pipeline import generate_wide_pool
        pool = generate_wide_pool(rule_num, 12000, seed)
        decisive = cc.other_rules_pass_mask(pool, rule_column)
        values = pd.to_numeric(pool[rule_column], errors="coerce")
        band_mask = decisive & (values > NEW_THRESHOLD) & (values <= 200.0)
        safe_mask = decisive & (values <= NEW_THRESHOLD)
        pool["band"] = np.where(band_mask, "band", np.where(safe_mask, "safe", "other"))
    elif version == "v3b2":
        from centroid_v3b2_pipeline import generate_noisy_pool
        pool = generate_noisy_pool(rule_num, 12000, seed)
        values = pd.to_numeric(pool[rule_column], errors="coerce")
        # v3b2 라벨 규칙: decisive 무관, 부채비율 값만 기준
        band_mask = (values > NEW_THRESHOLD) & (values <= 200.0)
        safe_mask = values <= NEW_THRESHOLD
        pool["band"] = np.where(band_mask, "band", np.where(safe_mask, "safe", "other"))
    else:
        raise ValueError(version)
    return pool


def run(args):
    start_time = datetime.now()
    run_dir = cc.CENTROID_ROOT / "result" / "centroid_matched" / f"edit_matched_{args.version}"
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{run_dir} 이미 존재. --overwrite 로 명시적으로 덮어쓰세요.")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    _, rule_column, theta, *_ = cc.rule_spec(args.rule)

    # ---------------- 1) layer 선정 (버전 일치 locate 결과) ----------------
    locate_dir = cc.CENTROID_ROOT / "result" / "centroid_matched" / f"locate_synth_{args.version}"
    report = json.load(open(locate_dir / f"rule{args.rule}_locate_report.json"))
    target_layer, cands, best_score = select_layer(report)
    print(f"[{args.version}] 선정 layer={target_layer} (debt_span 후보 {len(cands)}개, "
          f"min-flip 최대={best_score:.3f})")

    # ---------------- 2) delta_v (같은 locate 데이터의 캐시 hidden) ----------------
    cached_a = torch.load(locate_dir / "cache_A.pt")
    cached_b = torch.load(locate_dir / "cache_B.pt")
    delta_v = (cached_b["debt_span"][:, target_layer, :].mean(0)
               - cached_a["debt_span"][:, target_layer, :].mean(0))
    print(f"delta_v(={args.version} locate 쌍, L{target_layer}) norm={delta_v.norm().item():.3f}")

    tokenizer, model = cc.load_model(args.model_name, device=device)

    # ---------------- 3) gate (같은 버전 데이터, 같은 layer) ----------------
    pool = build_gate_pool(args.version, args.rule, args.seed)
    band_df = pool[pool["band"] == "band"].reset_index(drop=True)
    safe_df = pool[pool["band"] == "safe"].reset_index(drop=True)
    print(f"gate pool: band={len(band_df)} safe={len(safe_df)}")
    if len(band_df) == 0 or len(safe_df) == 0:
        raise RuntimeError("밴드/안전 샘플 부족")

    t0 = time.time()
    cache_path = run_dir / "cache_gate_hidden.pt"
    if cache_path.exists():
        # 캐시에 numpy 배열 포함 — 직접 생성한 로컬 파일이므로 weights_only=False 로드
        cached = torch.load(cache_path, weights_only=False)
        x_band, x_safe = cached["x_band"], cached["x_safe"]
    else:
        band_sub = band_df.sample(n=min(args.n_cal_extract, len(band_df)),
                                   random_state=args.seed).reset_index(drop=True)
        safe_sub = safe_df.sample(n=min(args.n_cal_extract, len(safe_df)),
                                   random_state=args.seed).reset_index(drop=True)
        hid_band, _, _ = cc.extract_layer_position_hidden(model, tokenizer, band_sub, rule_column,
                                                           device, batch_size=args.batch_size)
        hid_safe, _, _ = cc.extract_layer_position_hidden(model, tokenizer, safe_sub, rule_column,
                                                           device, batch_size=args.batch_size)
        x_band = hid_band["debt_span"][:, target_layer, :].numpy()
        x_safe = hid_safe["debt_span"][:, target_layer, :].numpy()
        torch.save({"x_band": x_band, "x_safe": x_safe}, cache_path)
    print(f"gate hidden 추출 완료 ({time.time()-t0:.1f}s)")

    X = np.concatenate([x_band, x_safe], axis=0)
    y = np.concatenate([np.ones(len(x_band)), np.zeros(len(x_safe))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed,
                                               stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료. 합성 held-out 정확도={gate_synth_held_acc:.4f}")

    # ---------------- 4) alpha 보정 (합성만) ----------------
    band_cal = band_df.sample(n=min(args.n_cal, len(band_df)), random_state=args.seed)
    safe_cal = safe_df.sample(n=min(args.n_cal, len(safe_df)), random_state=args.seed)

    def build_rows_info(sub_df):
        infos = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1]
                 for _, r in sub_df.iterrows()]
        batch = build_batch(tokenizer, infos, device)
        idxs = [idxs_for_position(info, "debt_span") for info in infos]
        return batch, idxs

    batch_band, idxs_band = build_rows_info(band_cal)
    batch_safe, idxs_safe = build_rows_info(safe_cal)
    with torch.no_grad():
        clean_band_pred = model(**batch_band).logits.argmax(-1).cpu().numpy()
        clean_safe_pred = model(**batch_safe).logits.argmax(-1).cpu().numpy()

    @torch.no_grad()
    def run_gated(batch, idxs_per_sample, alpha):
        module = get_layer_module(model, target_layer)
        hook = GatedShiftHook(idxs_per_sample, w_gate_t, b_gate, delta_v, alpha, ungated=False)
        handle = module.register_forward_hook(hook)
        try:
            out = model(**batch)
        finally:
            handle.remove()
        return out.logits.argmax(-1).cpu().numpy()

    cal_rows = []
    for alpha in args.alphas:
        pred_band = run_gated(batch_band, idxs_band, alpha)
        pred_safe = run_gated(batch_safe, idxs_safe, alpha)
        band_flip = (float((pred_band[clean_band_pred == 1] == 0).mean())
                     if (clean_band_pred == 1).any() else float("nan"))
        safe_false_flip = (float((pred_safe[clean_safe_pred == 1] == 0).mean())
                           if (clean_safe_pred == 1).any() else float("nan"))
        cal_rows.append({"alpha": alpha, "band_flip_rate": band_flip,
                         "safe_false_flip_rate": safe_false_flip})
        print(f"  alpha={alpha} | 밴드 flip={band_flip:.3f} 안전 오탐={safe_false_flip:.3f}")
    cal_df = pd.DataFrame(cal_rows)
    ok = cal_df[cal_df["safe_false_flip_rate"] <= args.max_safe_false_flip]
    best_alpha = float((ok if len(ok) else cal_df)
                       .sort_values("band_flip_rate", ascending=False).iloc[0]["alpha"])
    print(f"선정된 alpha: {best_alpha}")

    # ---------------- 5) test.csv 전체 평가 (여기서만 실데이터) ----------------
    print("*** test.csv를 여는 지점 (평가 전용) ***")
    df = cc.load_test_csv()
    decisive_real = cc.other_rules_pass_mask(df, rule_column)
    values_real = pd.to_numeric(df[rule_column], errors="coerce")
    band_mask_real = decisive_real & (values_real > NEW_THRESHOLD) & (values_real <= theta)
    safe_mask_real = decisive_real & (values_real <= NEW_THRESHOLD)
    cross_mask_real = ~decisive_real
    new_label = df["label"].copy()
    new_label.loc[band_mask_real] = 0

    module = get_layer_module(model, target_layer)

    @torch.no_grad()
    def predict_full(df_full, batch_size=64):
        preds = []
        rows = df_full.reset_index(drop=True)
        for i in range(0, len(rows), batch_size):
            sub = rows.iloc[i:i + batch_size]
            infos = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1]
                     for _, r in sub.iterrows()]
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

    from sklearn.metrics import accuracy_score, f1_score, recall_score
    labels = new_label.values
    full_result = {
        "accuracy": float(accuracy_score(labels, full_preds)),
        "macro_f1": float(f1_score(labels, full_preds, average="macro")),
        "recall_pass": float(recall_score(labels, full_preds, pos_label=1)),   # 통과(1) 재현율
        "recall_fail": float(recall_score(labels, full_preds, pos_label=0)),   # 탈락(0) 재현율
        "macro_recall": float(recall_score(labels, full_preds, average="macro")),
        "efficacy_band": float(accuracy_score(labels[band_mask_real.values],
                                               full_preds[band_mask_real.values])),
        "locality_safe": float(accuracy_score(labels[safe_mask_real.values],
                                               full_preds[safe_mask_real.values])),
        "locality_cross": float(accuracy_score(labels[cross_mask_real.values],
                                                full_preds[cross_mask_real.values])),
    }
    pd.DataFrame({"pred": full_preds}).to_csv(run_dir / f"rule{args.rule}_predictions.csv",
                                               index_label="index")
    print(f"[{args.version} matched edit 전체 평가] {full_result}")

    out = {
        "version": args.version, "rule": args.rule,
        "target_layer": target_layer, "layer_selection": {
            "rule": "debt_span 후보 중 min(flip 양방향) 최대, 동률이면 중앙값 층",
            "candidates": cands, "best_min_flip": best_score,
        },
        "locate_dir": str(locate_dir.relative_to(cc.CENTROID_ROOT)),
        "locate_data_md5": report.get("data_md5"),
        "delta_v_norm": float(delta_v.norm().item()),
        "gate_synth_heldout_acc": gate_synth_held_acc,
        "best_alpha": best_alpha, "calibration": cal_rows,
        "full_evaluation": full_result,
    }
    with open(run_dir / f"rule{args.rule}_matched_edit_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate, "delta_v": delta_v,
                "layer": target_layer, "best_alpha": best_alpha},
               run_dir / f"rule{args.rule}_matched_shift_config.pt")
    cc.write_manifest(run_dir, script="centroid_edit_matched.py", inputs=vars(args),
                       extra={"report_summary": full_result, "target_layer": target_layer},
                       start_time=start_time)
    print(f"저장 완료: {run_dir}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=VERSIONS)
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--n_cal_extract", type=int, default=1000)
    parser.add_argument("--n_cal", type=int, default=150)
    parser.add_argument("--max_safe_false_flip", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default=cc.HF_MODEL_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
