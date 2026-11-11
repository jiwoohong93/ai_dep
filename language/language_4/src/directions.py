"""Steering directions (diff-in-means, CAA), balanced sampling, and layer-wise probes."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROLE_OF = lambda m: np.where(m.pred_idx == m.unknown_option_idx, "unknown",
                             np.where(m.pred_idx == m.stereo_option_idx, "stereo", "anti"))


def add_pred_role(meta):
    meta = meta.copy()
    meta["pred_role"] = ROLE_OF(meta)
    return meta


def balanced_pos_neg(meta, pos_mask, neg_mask, cells=("category", "question_polarity"), seed=42, log=print):
    """Per cell, subsample pos/neg to min(n_pos, n_neg). Returns (pos_idx, neg_idx, cell_table)."""
    rng = np.random.RandomState(seed)
    pos_idx, neg_idx, rows = [], [], []
    for key, g in meta.groupby(list(cells)):
        p = g.index[pos_mask[g.index]].to_numpy()
        n = g.index[neg_mask[g.index]].to_numpy()
        k = min(len(p), len(n))
        rows.append(dict(zip(cells, key if isinstance(key, tuple) else (key,)), n_pos=len(p), n_neg=len(n), kept=k))
        if k == 0:
            continue
        pos_idx.extend(rng.choice(p, k, replace=False))
        neg_idx.extend(rng.choice(n, k, replace=False))
    table = pd.DataFrame(rows)
    empty = table[table.kept == 0]
    if len(empty):
        log(f"[balance] {len(empty)} empty cells excluded:\n{empty.to_string(index=False)}")
    return np.sort(pos_idx), np.sort(neg_idx), table


def diff_in_means(acts, pos_idx, neg_idx):
    """acts [L, n, d] -> r [L, d] = mean(pos) - mean(neg)."""
    p = acts[:, torch.as_tensor(pos_idx)].double().mean(1)
    n = acts[:, torch.as_tensor(neg_idx)].double().mean(1)
    return (p - n).float()


def layer_cos(a, b):
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1).numpy()


def probe_curve(acts_tr, y_tr, acts_va, y_va, C=1.0, seed=42):
    """Per-layer L2 logistic regression on standardized acts. acts [L, n, d]; y in {0,1}."""
    out = []
    for l in range(acts_tr.shape[0]):
        sc = StandardScaler().fit(acts_tr[l].numpy())
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(sc.transform(acts_tr[l].numpy()), y_tr)
        p = clf.predict_proba(sc.transform(acts_va[l].numpy()))[:, 1]
        out.append({"layer": l, "acc": float(((p > 0.5) == y_va).mean()), "auroc": float(roc_auc_score(y_va, p))})
    return pd.DataFrame(out)
