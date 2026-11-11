"""Phase 4: directions r_amb (behavior, CAA), r_disamb, layer-wise probe, benign residual norms."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

import numpy as np
import pandas as pd
import torch

from common import get_logger, load_config, run_dir, save_json, seed_all
from directions import add_pred_role, balanced_pos_neg, diff_in_means, layer_cos, probe_curve
from extract import load_layers, load_meta
from plotting import SERIES, plt, save

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
seed = cfg["seed"]
seed_all(seed)
A, S, R = run_dir(cfg, "acts"), run_dir(cfg, "steer"), run_dir(cfg, "results")
log = get_logger("directions", Path(cfg["paths"]["results"]) / "logs/04_directions.log")

meta = add_pred_role(load_meta(A / "kobbq_trainval"))
L = int(torch.load(sorted((A / "kobbq_trainval").glob("shard_*.pt"))[0], mmap=True)["acts"].shape[0])
layers = list(range(L))
acts = load_layers(A / "kobbq_trainval", layers, torch.bfloat16)  # [L, n, d]
log.info(f"acts {tuple(acts.shape)}")

tr_amb = meta[(meta.split == "train") & (meta.context_type == "ambig")]
log.info("train-ambig pred roles: " + str(tr_amb.pred_role.value_counts().to_dict()) +
         " (anti errors are excluded from D_pos/D_neg)")
pos_m = meta.pred_role.eq("unknown") & meta.index.isin(tr_amb.index)
neg_m = meta.pred_role.eq("stereo") & meta.index.isin(tr_amb.index)
pos, neg, tab = balanced_pos_neg(meta, pos_m, neg_m, seed=seed, log=log.info)
tab.to_csv(R / "balance_r_amb.csv", index=False)
log.info(f"r_amb behavior: D_pos={len(pos)} D_neg={len(neg)} (balanced per category x polarity)")
for nm, ix in (("pos", pos), ("neg", neg)):
    log.info(f"  letter of chosen option ({nm}): {meta.loc[ix, 'pred_idx'].value_counts().sort_index().to_dict()}")
r_beh = diff_in_means(acts, pos, neg)

# CAA: one (unknown, stereo) pair per train-ambig prompt -> balanced by construction
cu = load_layers(A / "caa_unknown", layers, torch.bfloat16)
cs = load_layers(A / "caa_stereo", layers, torch.bfloat16)
r_caa = (cu.double().mean(1) - cs.double().mean(1)).float()
del cu, cs

# r_disamb: disamb correct vs. stereo errors in counter-biased contexts (train)
tr_dis = meta[(meta.split == "train") & (meta.context_type == "disambig")]
pos_d = meta.index.isin(tr_dis.index) & (meta.pred_idx == meta.gold_idx)
neg_d = meta.index.isin(tr_dis.index) & meta.context_bias.eq("counter_biased") & meta.pred_role.eq("stereo")
pd_, nd_, tab_d = balanced_pos_neg(meta, pd.Series(pos_d, meta.index), pd.Series(neg_d, meta.index), seed=seed, log=log.info)
tab_d.to_csv(R / "balance_r_disamb.csv", index=False)
log.info(f"r_disamb: D_pos={len(pd_)} D_neg={len(nd_)}")
r_dis = diff_in_means(acts, pd_, nd_)

# mean residual norm per layer on benign-train (CAA scaling)
bmeta = load_meta(A / "benign")
btr = np.where(bmeta.split.values == "train")[0]
bacts = load_layers(A / "benign", layers, torch.bfloat16)
mean_norm = torch.stack([bacts[l, torch.as_tensor(btr)].float().norm(dim=-1).mean() for l in layers])
del bacts

cos_bc = layer_cos(r_beh, r_caa)
cos_ad = layer_cos(r_beh, r_dis)
torch.save({"r_behavior": r_beh, "r_caa": r_caa, "r_disamb": r_dis, "mean_norm": mean_norm,
            "n_pos": len(pos), "n_neg": len(neg)}, S / "directions.pt")

# probe: train on balanced train-ambig (unknown=1 vs stereo=0), eval on val-ambig templates
va = meta[(meta.split == "val") & (meta.context_type == "ambig") & meta.pred_role.isin(["unknown", "stereo"])]
tr_idx = np.concatenate([pos, neg]); y_tr = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
y_va = (va.pred_role == "unknown").to_numpy().astype(int)
probe = probe_curve(acts[:, torch.as_tensor(tr_idx)].float(), y_tr, acts[:, torch.as_tensor(va.index.to_numpy())].float(), y_va)
probe["r_norm_behavior"] = r_beh.norm(dim=-1).numpy()
probe["r_norm_caa"] = r_caa.norm(dim=-1).numpy()
probe["mean_norm_benign"] = mean_norm.numpy()
probe["cos_behavior_caa"] = cos_bc
probe["cos_amb_disamb"] = cos_ad
probe.to_csv(R / "directions_by_layer.csv", index=False)
log.info(f"val-ambig probe set: n={len(va)} unknown={y_va.sum()}\n{probe.round(3).to_string(index=False)}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(layers, probe.auroc, color=SERIES[0], label="AUROC")
ax[0].plot(layers, probe.acc, color=SERIES[1], label="accuracy")
ax[0].set(title="Probe: unknown vs stereo (train→val templates)", xlabel="layer", ylim=(0.4, 1.0))
ax[0].legend()
ax[1].plot(layers, cos_bc, color=SERIES[0], label="cos(r_behavior, r_CAA)")
ax[1].plot(layers, cos_ad, color=SERIES[1], label="cos(r_amb, r_disamb)")
ax[1].axhline(0, color="#a3a29d", lw=0.8)
ax[1].set(title="Direction cosine by layer", xlabel="layer", ylim=(-1, 1))
ax[1].legend()
save(fig, R / "directions_probe_cos.png")
save_json({"n_pos": len(pos), "n_neg": len(neg), "n_pos_dis": len(pd_), "n_neg_dis": len(nd_),
           "train_amb_roles": tr_amb.pred_role.value_counts().to_dict()}, R / "directions_summary.json")
