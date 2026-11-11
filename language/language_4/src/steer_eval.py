"""Build steering specs from saved artifacts and evaluate a condition on KoBBQ / KMMLU frames."""
import numpy as np
import torch

from eval_kmmlu import accuracy, group_acc, run_kmmlu
from eval_kobbq import metrics
from hooks import Steerer
from prompts import kobbq_prompt
from scoring import score_texts


def build_spec(cond, steer_dir, layers, r_kind="behavior", p=0.6, alpha=10.0, hb_tag="alphasteer"):
    """cond in {base, prompt, caa, alphasteer}; returns {layer: (kind, tensors)} or {}."""
    if cond in ("base", "prompt"):
        return {}
    if cond == "caa":
        D = torch.load(steer_dir / "directions.pt")
        r = D[f"r_{r_kind}"]
        return {l: ("caa", r[l] / r[l].norm() * D["mean_norm"][l]) for l in layers}
    if cond == "alphasteer":
        spec = {}
        for l in layers:
            f = torch.load(steer_dir / hb_tag / f"L{l}_p{p}_a{alpha}_{r_kind}.pt")
            spec[l] = ("alphasteer_lr", (f["w"], f["r"]))
        return spec
    raise ValueError(cond)


def eval_kobbq(model, tok, df, cfg, cond, spec, lam, max_tokens=24000):
    texts = [kobbq_prompt(tok, r, cfg, "prompt" if cond == "prompt" else "base") for r in df.to_dict("records")]
    with Steerer(model, spec, lam, cfg["alphasteer"]["mode"]):
        r = score_texts(model, tok, texts, "ABC", max_tokens)
    out = df.copy()
    out["pred_idx"], out["margin"] = r["pred_idx"], r["margin"]
    out["letter_logits"] = list(r["letter_logits"])
    return out


def eval_kmmlu(model, tok, df, shots, cfg, cond, spec, lam, max_tokens=24000):
    with Steerer(model, spec, lam, cfg["alphasteer"]["mode"]):
        return run_kmmlu(model, tok, df, shots, cfg, "prompt" if cond == "prompt" else "base", max_tokens)


def summarize(kb_pred, km_pred, km_base_acc):
    m = metrics(kb_pred)
    acc = accuracy(km_pred)
    m.update(kmmlu_acc=acc, kmmlu_abs_drop_pp=(km_base_acc - acc) * 100,
             kmmlu_rel_drop_pct=(km_base_acc - acc) / km_base_acc * 100,
             **{f"kmmlu_{g}": v for g, v in group_acc(km_pred).items()})
    return m


def screen_subset(kb_val, per_cell=8, seed=42):
    """Fixed stratified subset of KoBBQ val: <= per_cell samples per (template, version, context, polarity)."""
    rng = np.random.RandomState(seed)
    keep = []
    for _, idx in kb_val.groupby(["template_id", "version", "context_type", "question_polarity"]).indices.items():
        keep.extend(rng.choice(idx, min(per_cell, len(idx)), replace=False))
    return kb_val.iloc[sorted(keep)].reset_index(drop=True)
