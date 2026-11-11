"""Phase 5: AlphaSteer null-space projection P and steering matrix Delta for every layer / p / alpha / r.

Saves <steer>/<run>/alphasteer/L{l}_p{p}_a{alpha}_{rkind}.pt = {"w", "r"} with Delta (row conv.) M = w r^T,
plus <results>/<run>/alphasteer_diag.csv with null-space leakage, ||Delta h|| separation, conditioning.
H_b = benign-train last-token acts (KoBBQ disamb NOT included unless --hb_add_disamb).
H_m = train-ambig KoBBQ last-token acts (D_pos and D_neg and anti alike: all train ambig).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from alphasteer import benign_cov, null_basis, solve_tilde_delta
from common import get_logger, load_config, run_dir, seed_all
from extract import load_layer, load_meta

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--layers", default="all")
ap.add_argument("--hb_add_disamb", action="store_true", help="variant: add train-template KoBBQ disamb to H_b")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
cfg = load_config(args.config)
seed_all(cfg["seed"])
A, S, R = run_dir(cfg, "acts"), run_dir(cfg, "steer"), run_dir(cfg, "results")
tag = "alphasteer_hbdis" if args.hb_add_disamb else "alphasteer"
out = S / tag
out.mkdir(exist_ok=True)
log = get_logger("alphasteer", Path(cfg["paths"]["results"]) / f"logs/05_{tag}.log")
dev = torch.device(args.device)

D = torch.load(S / "directions.pt")
rk = {"behavior": D["r_behavior"], "caa": D["r_caa"]}
L = D["r_behavior"].shape[0]
layers = range(L) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
ac = cfg["alphasteer"]

km = load_meta(A / "kobbq_trainval")
bm = load_meta(A / "benign")
kmmlu = A / "kmmlu_val"
ix = lambda m: torch.as_tensor(np.where(m)[0])
i_tr_amb = ix((km.split == "train") & (km.context_type == "ambig"))
i_va_amb = ix((km.split == "val") & (km.context_type == "ambig"))
i_va_dis = ix((km.split == "val") & (km.context_type == "disambig"))
i_tr_dis = ix((km.split == "train") & (km.context_type == "disambig"))
i_b_tr, i_b_va = ix(bm.split == "train"), ix(bm.split == "val")
log.info(f"H_b benign-train={len(i_b_tr)}{' + train disamb=' + str(len(i_tr_dis)) if args.hb_add_disamb else ''} "
         f"| H_m train-ambig={len(i_tr_amb)}")

rows = []
for l in layers:
    t0 = time.time()
    kact = load_layer(A / "kobbq_trainval", l)
    bact = load_layer(A / "benign", l)
    Hb = [bact[i_b_tr]] + ([kact[i_tr_dis]] if args.hb_add_disamb else [])
    C, nb = benign_cov((x for h in Hb for x in h.split(2048)), bact.shape[1], dev)
    Hm = kact[i_tr_amb].to(dev, torch.float64)
    probes = {"benign_val": bact[i_b_va], "kobbq_val_amb": kact[i_va_amb], "kobbq_val_dis": kact[i_va_dis],
              "kmmlu_val": load_layer(kmmlu, l)}
    probes = {k: v.to(dev, torch.float64) for k, v in probes.items()}
    b_tr = bact[i_b_tr].to(dev, torch.float64)
    for p in ac["sweep_null_ratio"]:
        U, evals = null_basis(C, p)
        leak = lambda h: ((h @ U).norm(dim=1) / h.norm(dim=1))
        leak_tr, leak_va = leak(b_tr), leak(probes["benign_val"])
        for a in ac["sweep_ridge"]:
            for rname, rv in rk.items():
                r = rv[l].to(dev, torch.float64)
                w, rr, info = solve_tilde_delta(Hm, U, r, a)
                torch.save({"w": w.float().cpu(), "r": rr.float().cpu(), "layer": l, "p": p, "alpha": a,
                            "r_kind": rname, "n_b": nb}, out / f"L{l}_p{p}_a{a}_{rname}.pt")
                g = {k: (v @ w).abs() * r.norm() for k, v in probes.items()}  # ||Delta h||
                dm = (Hm @ w)[:, None] * r[None, :]
                rows.append({
                    "layer": l, "p": p, "alpha": a, "r_kind": rname, **info,
                    "leak_benign_train_mean": float(leak_tr.mean()), "leak_benign_train_max": float(leak_tr.max()),
                    "leak_benign_val_mean": float(leak_va.mean()), "leak_benign_val_max": float(leak_va.max()),
                    "cos_mean_Dh_m_r": float(torch.nn.functional.cosine_similarity(dm.mean(0), r, 0)),
                    "gain_train_amb_mean": float((Hm @ w).mean()),  # Delta h_m = gain * r; target gain = 1
                    "r_norm": float(r.norm()),
                    **{f"dh_{k}_median": float(v.median()) for k, v in g.items()},
                    "auroc_dh_amb_vs_benign": float(roc_auc_score(
                        np.r_[np.ones(len(g["kobbq_val_amb"])), np.zeros(len(g["benign_val"]))],
                        torch.cat([g["kobbq_val_amb"], g["benign_val"]]).cpu().numpy())),
                    "auroc_dh_amb_vs_kmmlu": float(roc_auc_score(
                        np.r_[np.ones(len(g["kobbq_val_amb"])), np.zeros(len(g["kmmlu_val"]))],
                        torch.cat([g["kobbq_val_amb"], g["kmmlu_val"]]).cpu().numpy())),
                })
    log.info(f"layer {l} done in {time.time() - t0:.0f}s | " + str({k: round(v, 4) for k, v in rows[-1].items()
                                                                  if isinstance(v, float)}))
    pd.DataFrame(rows).to_csv(R / f"{tag}_diag.csv", index=False)
