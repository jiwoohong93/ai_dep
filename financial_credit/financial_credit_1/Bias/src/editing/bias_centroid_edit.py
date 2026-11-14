"""Exp3+4-Bias: delta_v 계산 + gate 학습 + alpha 보정 + (gate 실데이터 검증만) real test 일부 확인.

위치: layer=9, marital-status + relationship 두 position 동시 (Exp2 causal patch로 확인된
"두 position을 같이 patch해야 효과가 남" 결과를 그대로 반영).

delta_v_marital-status = mu(married,qualified) - mu(unmarried,qualified) @ marital-status pos
delta_v_relationship   = mu(married,qualified) - mu(unmarried,qualified) @ relationship pos

gate: [marital-status_hidden ++ relationship_hidden](2048차원) -> "이 사람이 비혼으로 읽히는가"
      로지스틱 회귀, train.csv 유래 합성 married/unmarried 쌍으로만 학습 (test.csv 무사용).

alpha 보정: 합성 calibration set에서 비혼->기혼 flip률 vs 기혼 오탐(false positive shift)
           trade-off로 결정 (test.csv 무사용).

이 스크립트는 test.csv를 "gate 실데이터 정확도" 진단에 한해서만 읽는다(가중치 학습에는 전혀
안 씀) -- 최종 acc/DEO/efficacy 본평가는 별도 bias_centroid_evaluate.py(Exp5)에서 수행.
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
import bias_gen_pairs as bp  # noqa: E402
from src.editing.centroid_edit_keyselect import probe_weight_raw_space  # noqa: E402
from bias_centroid_hooks import (  # noqa: E402
    get_layer_module, build_batch_infos, build_batch, predict_batch,
)

POSITIONS = ["marital-status", "relationship"]
RUN_DIR_BASE = bc.RESULT_ROOT / "edit"


class GatedShiftBiasHook:
    """gate가 발화하면 marital-status/relationship 두 position에 각자의 delta_v를 동시 가산."""

    def __init__(self, idxs_marital, idxs_relationship, w_gate, b_gate,
                 delta_v_marital, delta_v_relationship, alpha):
        self.idxs_marital = idxs_marital
        self.idxs_relationship = idxs_relationship
        self.w_gate = w_gate
        self.b_gate = b_gate
        self.delta_v_marital = delta_v_marital
        self.delta_v_relationship = delta_v_relationship
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        w = self.w_gate.to(h.dtype).to(h.device)
        dm = (self.alpha * self.delta_v_marital).to(h.dtype).to(h.device)
        dr = (self.alpha * self.delta_v_relationship).to(h.dtype).to(h.device)
        for i in range(h.shape[0]):
            mar_vec = h[i, self.idxs_marital[i], :].mean(dim=0)
            rel_vec = h[i, self.idxs_relationship[i], :].mean(dim=0)
            feat = torch.cat([mar_vec, rel_vec])
            score = torch.dot(feat, w) + self.b_gate
            if score.item() > 0:
                h[i, self.idxs_marital[i], :] = h[i, self.idxs_marital[i], :] + dm
                h[i, self.idxs_relationship[i], :] = h[i, self.idxs_relationship[i], :] + dr
        return (h,) + tuple(output[1:]) if is_tuple else h


def run(args):
    start_time = datetime.now()
    RUN_DIR = RUN_DIR_BASE / f"layer{args.layer}_seed{args.seed}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    print(f"device={device} layer={args.layer} positions={POSITIONS}")

    train = bc.load_train_csv()
    tokenizer, model = bc.load_model(device=device)

    # ---------------- 1) delta_v (평균 크게, n=1000) ----------------
    t0 = time.time()
    married_dv, unmarried_dv, _ = bp.sample_marital_pairs(train, n=args.n_deltav, seed=args.seed)
    h_married, _, _ = bc.extract_multi_position_hidden(model, tokenizer, married_dv, device,
                                                         batch_size=args.batch_size, positions=POSITIONS)
    h_unmarried, _, _ = bc.extract_multi_position_hidden(model, tokenizer, unmarried_dv, device,
                                                           batch_size=args.batch_size, positions=POSITIONS)
    delta_v = {}
    for pos in POSITIONS:
        mu_married = h_married[pos][:, args.layer, :].mean(dim=0)
        mu_unmarried = h_unmarried[pos][:, args.layer, :].mean(dim=0)
        delta_v[pos] = mu_married - mu_unmarried
        print(f"delta_v[{pos}] norm={delta_v[pos].norm().item():.3f} (n={args.n_deltav})")
    print(f"delta_v 계산 완료 ({time.time()-t0:.1f}s)")

    # ---------------- 2) gate 학습 (합성, train.csv 유래, 넓은 문맥) ----------------
    t0 = time.time()
    married_g, unmarried_g, _ = bp.sample_marital_pairs(train, n=args.n_gate, seed=args.seed + 1)
    h_married_g, _, _ = bc.extract_multi_position_hidden(model, tokenizer, married_g, device,
                                                           batch_size=args.batch_size, positions=POSITIONS)
    h_unmarried_g, _, _ = bc.extract_multi_position_hidden(model, tokenizer, unmarried_g, device,
                                                             batch_size=args.batch_size, positions=POSITIONS)
    X_married = torch.cat([h_married_g["marital-status"][:, args.layer, :],
                            h_married_g["relationship"][:, args.layer, :]], dim=1).numpy()
    X_unmarried = torch.cat([h_unmarried_g["marital-status"][:, args.layer, :],
                              h_unmarried_g["relationship"][:, args.layer, :]], dim=1).numpy()
    X = np.concatenate([X_unmarried, X_married], axis=0)  # label 1 = 비혼(발화 대상)
    y = np.concatenate([np.ones(len(X_unmarried)), np.zeros(len(X_married))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    w_gate, b_gate = probe_weight_raw_space(X_tr, y_tr)
    w_gate_t = torch.tensor(w_gate, dtype=torch.float32)
    gate_synth_held_acc = float((((X_te @ w_gate + b_gate) > 0).astype(int) == y_te).mean())
    print(f"gate 학습 완료 (n_unmarried={len(X_unmarried)} n_married={len(X_married)}) "
          f"합성 held-out acc={gate_synth_held_acc:.4f} ({time.time()-t0:.1f}s)")

    # ---------------- 3) alpha 보정 (합성 calibration, test.csv 무사용) ----------------
    module = get_layer_module(model, args.layer)
    married_cal, unmarried_cal, _ = bp.sample_marital_pairs(train, n=args.n_cal, seed=args.seed + 2)

    def build_rows(sub_df):
        texts, infos = build_batch_infos(tokenizer, sub_df)
        batch = build_batch(tokenizer, infos, device)
        idxs_m = [info["marital-status"] for info in infos]
        idxs_r = [info["relationship"] for info in infos]
        return batch, idxs_m, idxs_r

    batch_um, idxs_m_um, idxs_r_um = build_rows(unmarried_cal)
    batch_mr, idxs_m_mr, idxs_r_mr = build_rows(married_cal)
    clean_pred_um = predict_batch(model, batch_um)
    clean_pred_mr = predict_batch(model, batch_mr)
    print(f"calibration baseline: unmarried P(1)={clean_pred_um.mean():.4f} "
          f"married P(1)={clean_pred_mr.mean():.4f}")

    @torch.no_grad()
    def run_gated(batch, idxs_m, idxs_r, alpha):
        hook = GatedShiftBiasHook(idxs_m, idxs_r, w_gate_t, b_gate, delta_v["marital-status"],
                                   delta_v["relationship"], alpha)
        handle = module.register_forward_hook(hook)
        try:
            out = model(**batch)
        finally:
            handle.remove()
        return out.logits.argmax(-1).cpu().numpy()

    cal_rows = []
    for alpha in args.alphas:
        pred_um = run_gated(batch_um, idxs_m_um, idxs_r_um, alpha)
        pred_mr = run_gated(batch_mr, idxs_m_mr, idxs_r_mr, alpha)
        unmarried_flip = float(((clean_pred_um == 0) & (pred_um == 1)).sum() / max((clean_pred_um == 0).sum(), 1))
        married_false_flip = float(((clean_pred_mr == 1) & (pred_mr == 0)).sum() / max((clean_pred_mr == 1).sum(), 1))
        cal_rows.append({"alpha": alpha, "unmarried_flip_0to1": unmarried_flip,
                          "married_false_flip_1to0": married_false_flip})
        print(f"  alpha={alpha} | 비혼 flip(0->1)={unmarried_flip:.3f} 기혼 오탐(1->0)={married_false_flip:.3f}")

    with open(RUN_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal_rows, f, ensure_ascii=False, indent=2)

    # ---------------- 4) gate 실데이터 정확도 (진단용, test.csv 일부 -- 학습에는 미사용) ----------------
    test = bc.load_test_csv()
    male_or_female = test["sex"].isin(["Male", "Female"])
    unmarried_mask = test["marital-status"].isin(["Never-married", "Divorced", "Separated", "Widowed"])
    real_unmarried = test[male_or_female & unmarried_mask].sample(
        n=min(300, (male_or_female & unmarried_mask).sum()), random_state=args.seed)
    real_married = test[male_or_female & ~unmarried_mask].sample(
        n=min(300, (male_or_female & ~unmarried_mask).sum()), random_state=args.seed)
    h_ru, _, _ = bc.extract_multi_position_hidden(model, tokenizer, real_unmarried, device,
                                                    batch_size=args.batch_size, positions=POSITIONS)
    h_rm, _, _ = bc.extract_multi_position_hidden(model, tokenizer, real_married, device,
                                                    batch_size=args.batch_size, positions=POSITIONS)
    Xr_u = torch.cat([h_ru["marital-status"][:, args.layer, :], h_ru["relationship"][:, args.layer, :]], dim=1).numpy()
    Xr_m = torch.cat([h_rm["marital-status"][:, args.layer, :], h_rm["relationship"][:, args.layer, :]], dim=1).numpy()
    gate_pred = np.concatenate([((Xr_u @ w_gate + b_gate) > 0).astype(int),
                                 ((Xr_m @ w_gate + b_gate) > 0).astype(int)])
    gate_true = np.concatenate([np.ones(len(Xr_u)), np.zeros(len(Xr_m))])
    gate_real_acc = float((gate_pred == gate_true).mean())
    print(f"gate 실데이터(test.csv 일부, 진단 전용) 정확도={gate_real_acc:.4f}")

    report = {
        "layer": args.layer, "positions": POSITIONS,
        "n_deltav": args.n_deltav, "n_gate": args.n_gate, "n_cal": args.n_cal,
        "delta_v_norm": {p: float(delta_v[p].norm().item()) for p in POSITIONS},
        "gate_synth_heldout_acc": gate_synth_held_acc, "gate_real_acc": gate_real_acc,
        "calibration": cal_rows,
    }
    with open(RUN_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save({"w_gate": w_gate_t, "b_gate": b_gate,
                "delta_v_marital": delta_v["marital-status"], "delta_v_relationship": delta_v["relationship"],
                "layer": args.layer}, RUN_DIR / "shift_config.pt")

    bc.write_manifest(RUN_DIR, script="bias_centroid_edit.py", inputs=vars(args),
                       extra={"report_summary": {"gate_synth_heldout_acc": gate_synth_held_acc,
                                                  "gate_real_acc": gate_real_acc}}, start_time=start_time)
    print(f"저장 완료: {RUN_DIR}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--n_deltav", type=int, default=1000)
    parser.add_argument("--n_gate", type=int, default=1000)
    parser.add_argument("--n_cal", type=int, default=200)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
