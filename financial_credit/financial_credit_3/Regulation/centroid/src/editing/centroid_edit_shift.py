"""Exp 4(a) (EDRM centroid): gated shift hook (inference-time).

L21(hidden_states 인덱스) debt_span 표현에 대해:
  gate: 밴드(85~200) probe — 생성 샘플로 학습, 실데이터 밴드/안전 샘플로 held-out 검증
        (생성->실데이터 도메인 갭 확인)
  hook: gate 양성 샘플에만 h += alpha * delta_v  (delta_v = L21의 mu_B - mu_A, 구 rule 기준,
        Exp1/2에서 계산한 실측 centroid 차)

alpha in {0.5, 1, 2} 스윕, calibration은 생성 샘플만 사용 (실데이터는 평가 전용).
ungated 버전(전체 +alpha*delta_v)도 대조군으로 비교해 gate의 기여를 분리 측정한다.
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
from centroid_causal_patch import get_layer_module, build_batch, idxs_for_position  # noqa: E402
from centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402

TARGET_LAYER = 21  # hidden_states 인덱스 (== encoder.layer[20] 출력)


class GatedShiftHook:
    """gate(raw-space linear: w_gate·h + b_gate > 0)가 양성인 샘플의 debt_span 위치에만
    alpha*delta_v를 가산. ungated=True면 게이트 무시하고 전체에 적용."""

    def __init__(self, idxs_per_sample, w_gate, b_gate, delta_v, alpha, ungated=False):
        self.idxs_per_sample = idxs_per_sample
        self.w_gate = w_gate
        self.b_gate = b_gate
        self.delta_v = delta_v
        self.alpha = alpha
        self.ungated = ungated
        self.gate_fired = None  # 마지막 호출의 gate 발동 여부 (분석용)

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        w = self.w_gate.to(h.dtype).to(h.device)
        b = self.b_gate
        d = (self.alpha * self.delta_v).to(h.dtype).to(h.device)
        fired = []
        for i, idxs in enumerate(self.idxs_per_sample):
            span_vec = h[i, idxs, :].mean(dim=0)  # gate는 clean(가산 전) 값으로 판단
            gate_on = self.ungated or (torch.dot(span_vec, w) + b).item() > 0
            fired.append(bool(gate_on))
            if gate_on:
                h[i, idxs, :] = h[i, idxs, :] + d
        self.gate_fired = fired
        return (h,) + tuple(output[1:]) if is_tuple else h


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


def build_rows_info(tokenizer, sub_df, rule_column, device):
    infos = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in sub_df.iterrows()]
    batch = build_batch(tokenizer, infos, device)
    idxs = [idxs_for_position(info, "debt_span") for info in infos]
    return infos, batch, idxs


def run(args):
    cc.ensure_dirs()
    out_dir = cc.RESULT_ROOT / "edit" / "shift"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, rule_column, theta, *_ = cc.rule_spec(args.rule)
    device = f"cuda:{args.gpu_id}"
    tokenizer, model = cc.load_model(args.model_name, device=device)

    # ---------------- gate 학습 (생성 샘플) ----------------
    synth = pd.read_csv(cc.RESULT_ROOT / "gen_samples" / f"rule{args.rule}_synthetic_samples.csv")
    span_cache = torch.load(cc.CACHE_DIR / f"gen_samples_rule{args.rule}_hidden.pt")["debt_span"]
    x_all = span_cache[:, TARGET_LAYER, :].numpy()
    band_mask = (synth["band"] == "band").values
    safe_mask = (synth["band"] == "safe").values

    X = np.concatenate([x_all[band_mask], x_all[safe_mask]], axis=0)
    y = np.concatenate([np.ones(band_mask.sum()), np.zeros(safe_mask.sum())])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료. 생성 held-out 정확도={gate_synth_held_acc:.4f}")

    # ---------------- delta_v (실측 centroid 차, Exp1/2 캐시 재사용) ----------------
    df = cc.load_test_csv()
    real_a, real_b = cc.decisive_split(df, args.rule, n_per_group=500, seed=0)
    cache_a = cc.CACHE_DIR / f"locate_scan_rule{args.rule}_A.pt"
    cache_b = cc.CACHE_DIR / f"locate_scan_rule{args.rule}_B.pt"
    _, span_a_real = cc.get_cached_hidden(cache_a, model, tokenizer, real_a, rule_column, device)
    _, span_b_real = cc.get_cached_hidden(cache_b, model, tokenizer, real_b, rule_column, device)
    delta_v = span_b_real[:, TARGET_LAYER, :].mean(0) - span_a_real[:, TARGET_LAYER, :].mean(0)
    print(f"delta_v norm={delta_v.norm().item():.3f}")

    # ---------------- 실데이터 gate 도메인 갭 검증 ----------------
    rng = np.random.default_rng(args.seed)
    real_band_idx = rng.choice(df.index[cc.real_band_mask(df, args.rule, args.new_threshold)],
                                size=args.n_real_gate_check, replace=False)
    real_safe_mask = cc.other_rules_pass_mask(df, rule_column) & (
        pd.to_numeric(df[rule_column], errors="coerce") <= args.new_threshold)
    real_safe_idx = rng.choice(df.index[real_safe_mask], size=args.n_real_gate_check, replace=False)
    real_band_df = df.loc[real_band_idx].reset_index(drop=True)
    real_safe_df = df.loc[real_safe_idx].reset_index(drop=True)

    cache_rb = cc.CACHE_DIR / f"gatecheck_rule{args.rule}_real_band.pt"
    cache_rs = cc.CACHE_DIR / f"gatecheck_rule{args.rule}_real_safe.pt"
    _, span_real_band = cc.get_cached_hidden(cache_rb, model, tokenizer, real_band_df, rule_column, device)
    _, span_real_safe = cc.get_cached_hidden(cache_rs, model, tokenizer, real_safe_df, rule_column, device)
    x_real_band = span_real_band[:, TARGET_LAYER, :].numpy()
    x_real_safe = span_real_safe[:, TARGET_LAYER, :].numpy()
    real_gate_pred = np.concatenate([
        ((x_real_band @ w_gate + b_gate) > 0).astype(int),
        ((x_real_safe @ w_gate + b_gate) > 0).astype(int),
    ])
    real_gate_true = np.concatenate([np.ones(len(x_real_band)), np.zeros(len(x_real_safe))])
    gate_real_acc = float((real_gate_pred == real_gate_true).mean())
    print(f"gate 실데이터 정확도(도메인 갭 확인)={gate_real_acc:.4f} "
          f"(생성 held-out {gate_synth_held_acc:.4f} 대비 Δ={gate_real_acc-gate_synth_held_acc:+.4f})")

    # ---------------- alpha 스윕 (calibration=생성 샘플만) ----------------
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
            rows.append({
                "alpha": alpha, "gated": gated,
                "band_flip_rate": band_flip, "safe_false_flip_rate": safe_false_flip,
                "gate_fire_rate_band": float(fired_band.mean()), "gate_fire_rate_safe": float(fired_safe.mean()),
            })
            print(f"  alpha={alpha} gated={gated} | 밴드 flip={band_flip:.3f} "
                  f"안전 오탐-flip={safe_false_flip:.3f} | gate발동률(밴드/안전)={fired_band.mean():.2f}/{fired_safe.mean():.2f}")

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(out_dir / f"rule{args.rule}_alpha_calibration.csv", index=False)

    gated_rows = cal_df[cal_df["gated"]]
    gated_rows = gated_rows[gated_rows["safe_false_flip_rate"] <= args.max_safe_false_flip]
    if len(gated_rows):
        best_alpha = float(gated_rows.sort_values("band_flip_rate", ascending=False).iloc[0]["alpha"])
    else:
        best_alpha = float(cal_df[cal_df["gated"]].sort_values("band_flip_rate", ascending=False).iloc[0]["alpha"])
    print(f"\n선정된 alpha (안전 오탐 <= {args.max_safe_false_flip} 제약 내 밴드 flip 최대): {best_alpha}")

    report = {
        "rule": args.rule, "layer": TARGET_LAYER,
        "gate_synth_heldout_acc": gate_synth_held_acc, "gate_real_domain_acc": gate_real_acc,
        "gate_domain_gap": gate_real_acc - gate_synth_held_acc,
        "delta_v_norm": float(delta_v.norm().item()),
        "best_alpha": best_alpha,
        "calibration": rows,
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
    parser.add_argument("--new_threshold", type=float, default=85.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--n_cal", type=int, default=150)
    parser.add_argument("--n_real_gate_check", type=int, default=150)
    parser.add_argument("--max_safe_false_flip", type=float, default=0.02)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default=cc.LOCAL_MODEL_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
