"""Exp3+4-cg: rule(정책) 편집 -- capital-gain 임계값 7,000 -> 7,700.

위치: layer=15, capital-gain position (Exp1 스캔 재사용으로 확정: cg 신호 강함 + marital 오염
최소). delta_v = mu(cg<7000, 발동 안 함) - mu(band 7000~7700, 구rule 발동) -- band를
"발동 안 함" 방향으로 밀어주는 벡터.

gate 학습 데이터: Regulation v3a 레시피 그대로 -- gaussian(새 band 중심) + train.csv 실제
문맥으로 다양성 확보 (won의 generate_cg_gaussian_train.py 철학을 새 band에 맞게 재사용:
고정 프로필 대신 실제 train 행 위에 cg만 gaussian으로 얹어 문맥 다양성을 넓힘 -- v1의 좁은
합성 데이터 문제 재발 방지, Regulation v3a/v3b2에서 검증된 접근).

band = [7000,7700) : gate가 발화해야 함
non-band = cg<7000 또는 cg>=7700 : gate가 발화하면 안 됨
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


import bias_centroid_common as bc  # noqa: E402
from src.editing.centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402
from bias_centroid_hooks import (  # noqa: E402
    get_layer_module, build_batch_infos, build_batch, make_patch_hook, predict_batch,
)

POSITION = "capital-gain"
OLD_THETA = 7000.0
NEW_THETA = 7700.0
RUN_DIR_BASE = bc.RESULT_ROOT / "edit_cg"


class GatedShiftCgHook:
    def __init__(self, idxs, w_gate, b_gate, delta_v, alpha):
        self.idxs = idxs
        self.w_gate = w_gate
        self.b_gate = b_gate
        self.delta_v = delta_v
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        w = self.w_gate.to(h.dtype).to(h.device)
        d = (self.alpha * self.delta_v).to(h.dtype).to(h.device)
        for i, idxs in enumerate(self.idxs):
            vec = h[i, idxs, :].mean(dim=0)
            score = torch.dot(vec, w) + self.b_gate
            if score.item() > 0:
                h[i, idxs, :] = h[i, idxs, :] + d
        return (h,) + tuple(output[1:]) if is_tuple else h


def sample_gaussian_wide_cg_pool(train_df, n_rows, seed):
    """generate_cg_gaussian_train.py 철학(gaussian around new band) + 실제 문맥 다양성."""
    rng = np.random.default_rng(seed)
    base = train_df.sample(n=n_rows, random_state=seed, replace=len(train_df) < n_rows).reset_index(drop=True)
    center = (OLD_THETA + NEW_THETA) / 2  # 7350
    half_width = (NEW_THETA - OLD_THETA) / 2  # 350
    sigma = half_width  # band 폭과 비슷한 스케일로 -- band 안팎 골고루 샘플되게
    cg = rng.normal(center, sigma * 3, n_rows)  # 넓게(3sigma) 흔들어 band 밖도 충분히 포함
    cg = np.clip(np.round(cg), 0, None).astype(int)
    base["capital-gain"] = cg
    base["_band"] = (cg >= OLD_THETA) & (cg < NEW_THETA)
    return base


def run(args):
    start_time = datetime.now()
    RUN_DIR = RUN_DIR_BASE / f"layer{args.layer}_seed{args.seed}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    print(f"device={device} layer={args.layer} position={POSITION} band=[{OLD_THETA},{NEW_THETA})")

    train = bc.load_train_csv()
    tokenizer, model = bc.load_model(device=device)

    # ---------------- 1) delta_v ----------------
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    base_dv = train.sample(n=args.n_deltav, random_state=args.seed).reset_index(drop=True)
    band_dv = base_dv.copy()
    band_dv["capital-gain"] = rng.integers(int(OLD_THETA), int(NEW_THETA), size=len(base_dv))
    below_dv = base_dv.copy()
    below_dv["capital-gain"] = rng.integers(0, int(OLD_THETA), size=len(base_dv))

    h_band, _, _ = bc.extract_multi_position_hidden(model, tokenizer, band_dv, device,
                                                      batch_size=args.batch_size, positions=[POSITION])
    h_below, _, _ = bc.extract_multi_position_hidden(model, tokenizer, below_dv, device,
                                                       batch_size=args.batch_size, positions=[POSITION])
    delta_v = h_below[POSITION][:, args.layer, :].mean(dim=0) - h_band[POSITION][:, args.layer, :].mean(dim=0)
    print(f"delta_v norm={delta_v.norm().item():.3f} (n={args.n_deltav}) ({time.time()-t0:.1f}s)")

    # ---------------- 2) gate 학습 (gaussian + 실제 문맥 다양성) ----------------
    t0 = time.time()
    pool = sample_gaussian_wide_cg_pool(train, args.n_gate, args.seed + 1)
    print(f"gate pool: band={pool['_band'].sum()} non-band={(~pool['_band']).sum()}")
    h_pool, _, _ = bc.extract_multi_position_hidden(model, tokenizer, pool, device,
                                                      batch_size=args.batch_size, positions=[POSITION])
    X = h_pool[POSITION][:, args.layer, :].numpy()
    y = pool["_band"].to_numpy().astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료 합성 held-out acc={gate_synth_held_acc:.4f} ({time.time()-t0:.1f}s)")

    # ---------------- 3) alpha 보정 (합성 calibration) ----------------
    module = get_layer_module(model, args.layer)
    band_cal = train.sample(n=args.n_cal, random_state=args.seed + 2).reset_index(drop=True)
    band_cal["capital-gain"] = rng.integers(int(OLD_THETA), int(NEW_THETA), size=len(band_cal))
    nonband_cal = train.sample(n=args.n_cal, random_state=args.seed + 3).reset_index(drop=True)
    nonband_cal["capital-gain"] = rng.integers(0, int(OLD_THETA), size=len(nonband_cal))

    def build_rows(sub_df):
        _, infos = build_batch_infos(tokenizer, sub_df)
        batch = build_batch(tokenizer, infos, device)
        idxs = [info[POSITION] for info in infos]
        return batch, idxs

    batch_band, idxs_band = build_rows(band_cal)
    batch_nb, idxs_nb = build_rows(nonband_cal)
    clean_band = predict_batch(model, batch_band)
    clean_nb = predict_batch(model, batch_nb)
    print(f"calibration baseline: band P(1)={clean_band.mean():.4f} non-band(below) P(1)={clean_nb.mean():.4f}")

    @torch.no_grad()
    def run_gated(batch, idxs, alpha):
        hook = GatedShiftCgHook(idxs, w_gate_t, b_gate, delta_v, alpha)
        handle = module.register_forward_hook(hook)
        try:
            out = model(**batch)
        finally:
            handle.remove()
        return out.logits.argmax(-1).cpu().numpy()

    cal_rows = []
    for alpha in args.alphas:
        pred_band = run_gated(batch_band, idxs_band, alpha)
        pred_nb = run_gated(batch_nb, idxs_nb, alpha)
        band_flip = float(((clean_band == 1) & (pred_band == 0)).sum() / max((clean_band == 1).sum(), 1))
        nb_false_flip = float(((clean_nb == 1) & (pred_nb == 0)).sum() / max((clean_nb == 1).sum(), 1))
        cal_rows.append({"alpha": alpha, "band_flip_1to0": band_flip, "nonband_false_flip_1to0": nb_false_flip})
        print(f"  alpha={alpha} | band flip(1->0)={band_flip:.3f} non-band 오탐={nb_false_flip:.3f}")

    report = {"layer": args.layer, "position": POSITION, "old_theta": OLD_THETA, "new_theta": NEW_THETA,
              "n_deltav": args.n_deltav, "n_gate": args.n_gate, "delta_v_norm": float(delta_v.norm().item()),
              "gate_synth_heldout_acc": gate_synth_held_acc, "calibration": cal_rows}
    with open(RUN_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate, "delta_v": delta_v, "layer": args.layer,
                "position": POSITION}, RUN_DIR / "shift_config.pt")
    bc.write_manifest(RUN_DIR, script="bias_centroid_edit_cg.py", inputs=vars(args),
                       extra={"gate_synth_heldout_acc": gate_synth_held_acc}, start_time=start_time)
    print(f"저장 완료: {RUN_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--n_deltav", type=int, default=1000)
    parser.add_argument("--n_gate", type=int, default=2000)
    parser.add_argument("--n_cal", type=int, default=200)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
