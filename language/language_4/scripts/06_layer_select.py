"""Phase 6a: layer candidates from ||Delta h|| separation (benign-val vs KoBBQ val-ambig), reported with probe.

Rule (fixed before any steering run): at the default (p, alpha), per r kind, rank layers 2..L-3 by
AUROC(||Delta h||: KoBBQ val-ambig vs benign-val), ties (AUROC rounded to 3 decimals; it saturates at 1.0 because
the KoBBQ prompt format alone separates the sets) broken by selectivity = median ||Dh|| (val-ambig) / p95 ||Dh||
(benign-val). Candidates = top-4 single layers + contiguous groups of 2/3/4 layers around the best layer b:
[b,b+1], [b-1,b+1], [b-1,b+2].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

import numpy as np
import pandas as pd
import torch

from common import get_logger, load_config, run_dir, save_json
from extract import load_layer, load_meta
from plotting import SERIES, plt, save

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
R = run_dir(cfg, "results")
log = get_logger("layers", Path(cfg["paths"]["results"]) / "logs/06_layer_select.log")
ac = cfg["alphasteer"]
d = pd.read_csv(R / "alphasteer_diag.csv")
probe = pd.read_csv(R / "directions_by_layer.csv")
L = int(probe.layer.max()) + 1
d0 = d[(d.p == ac["null_ratio"]) & (d.alpha == ac["ridge"])].copy()

# selectivity from the saved w (||Dh|| = |h.w| ||r||, so ||r|| cancels in the ratio)
A, S = run_dir(cfg, "acts"), run_dir(cfg, "steer")
km, bm = load_meta(A / "kobbq_trainval"), load_meta(A / "benign")
i_amb = torch.as_tensor(np.where((km.split == "val") & (km.context_type == "ambig"))[0])
i_dis = torch.as_tensor(np.where((km.split == "val") & (km.context_type == "disambig"))[0])
i_bva = torch.as_tensor(np.where(bm.split == "val")[0])
sel_rows = []
for l in sorted(d0.layer.unique()):
    k, b, q = load_layer(A / "kobbq_trainval", l), load_layer(A / "benign", l)[i_bva], load_layer(A / "kmmlu_val", l)
    for rk in d0.r_kind.unique():
        w = torch.load(S / "alphasteer" / f"L{l}_p{ac['null_ratio']}_a{ac['ridge']}_{rk}.pt")["w"]
        g = lambda x: (x @ w).abs().numpy()
        ga, gd, gb, gq = g(k[i_amb]), g(k[i_dis]), g(b), g(q)
        sel_rows.append({"layer": l, "r_kind": rk, "selectivity": np.median(ga) / np.percentile(gb, 95),
                         "selectivity_kmmlu": np.median(ga) / np.percentile(gq, 95),
                         "amb_over_dis_median": np.median(ga) / np.median(gd)})
d0 = d0.merge(pd.DataFrame(sel_rows), on=["layer", "r_kind"])
d0.to_csv(R / "layer_selection_table.csv", index=False)

cands = {}
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, (rk, g) in enumerate(d0.groupby("r_kind")):
    g = g.set_index("layer").sort_index()
    ax.plot(g.index, g.auroc_dh_amb_vs_benign, color=SERIES[i], label=f"‖Δh‖ AUROC amb vs benign ({rk})")
    ok = g.loc[2:L - 3].assign(auroc_r=lambda x: x.auroc_dh_amb_vs_benign.round(3)) \
        .sort_values(["auroc_r", "selectivity"], ascending=False)
    b = int(ok.index[0])
    singles = [[int(x)] for x in ok.index[:4]]
    groups = [list(range(b, b + 2)), list(range(b - 1, b + 2)), list(range(b - 1, b + 3))]
    groups = [[x for x in gr if 0 <= x < L] for gr in groups]
    cands[rk] = singles + [gr for gr in groups if gr not in singles]
    log.info(f"[{rk}] by layer:\n{g[['auroc_dh_amb_vs_benign', 'auroc_dh_amb_vs_kmmlu', 'selectivity', 'selectivity_kmmlu', 'amb_over_dis_median', 'leak_benign_val_mean', 'gain_train_amb_mean']].round(3).to_string()}")
    log.info(f"[{rk}] candidates: {cands[rk]}")
ax.plot(probe.layer, probe.auroc, color=SERIES[2], label="probe AUROC (unknown vs stereo)")
ax.set(xlabel="layer", ylabel="AUROC", ylim=(0.4, 1.01), title=f"Layer selection (p={ac['null_ratio']}, α={ac['ridge']})")
ax.legend(fontsize=8)
save(fig, R / "layer_selection.png")
save_json(cands, R / "layer_candidates.json")
