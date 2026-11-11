"""KMMLU loading, KMMLU-val construction and accuracy/drop metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# lm-evaluation-harness kmmlu/direct supercategory tags (official KMMLU grouping)
SUPERCAT = {
    "stem": ["Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering", "Computer-Science", "Ecology",
             "Electrical-Engineering", "Information-Technology", "Materials-Engineering", "math",
             "Mechanical-Engineering"],
    "applied_science": ["Aviation-Engineering-and-Maintenance", "Electronics-Engineering", "Energy-Management",
                        "Environmental-Science", "Gas-Technology-and-Engineering", "Geomatics", "Industrial-Engineer",
                        "Machine-Design-and-Manufacturing", "Maritime-Engineering", "Nondestructive-Testing",
                        "Railway-and-Automotive-Engineering", "Telecommunications-and-Wireless-Technology"],
    "humss": ["Accounting", "Criminal-Law", "Economics", "Education", "korean-history", "Law", "Management",
              "Political-Science-and-Sociology", "Psychology", "Social-Welfare", "Taxation"],
    "other": ["Agricultural-Sciences", "Construction", "Fashion", "Food-Processing", "Health",
              "Interior-Architecture-and-Design", "Marketing", "Patent", "Public-Safety", "Real-Estate",
              "Refrigerating-Machinery"],
}
SUBJECT2GROUP = {s: g for g, ss in SUPERCAT.items() for s in ss}


def load_split(raw, split):
    d = Path(raw) / "HAERAE-HUB__KMMLU/data"
    frames = []
    for f in sorted(d.glob(f"*-{split}.csv")):
        subj = f.name[: -len(f"-{split}.csv")]
        x = pd.read_csv(f)
        x["subject"] = subj
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    assert set(df.subject) == set(SUBJECT2GROUP), set(df.subject) ^ set(SUBJECT2GROUP)
    assert df.answer.between(1, 4).all()
    df["group"] = df.subject.map(SUBJECT2GROUP)
    df["qid"] = [f"{split}-{s}-{i}" for s, i in zip(df.subject, df.groupby("subject").cumcount())]
    return df


def make_val(train, size=2000, seed=42):
    """Subject-stratified fixed subset of the KMMLU train split (equal per subject, remainder random)."""
    rng = np.random.RandomState(seed)
    subjects = sorted(train.subject.unique())
    per = size // len(subjects)
    picks = []
    for s in subjects:
        idx = np.where(train.subject.values == s)[0]
        picks.extend(rng.choice(idx, min(per, len(idx)), replace=False))
    rest = np.setdiff1d(np.arange(len(train)), picks)
    picks.extend(rng.choice(rest, size - len(picks), replace=False))
    return train.iloc[sorted(picks)].reset_index(drop=True)


def shots_by_subject(dev, k):
    return {s: g.head(k).to_dict("records") for s, g in dev.groupby("subject")}


def accuracy(pred_df):
    return float((pred_df.pred_idx == pred_df.answer - 1).mean())


def group_acc(pred_df):
    c = pred_df.assign(correct=(pred_df.pred_idx == pred_df.answer - 1))
    return c.groupby("group").correct.mean().to_dict()


def drop(acc_base, acc_steer):
    return {"abs_drop_pp": (acc_base - acc_steer) * 100, "rel_drop_pct": (acc_base - acc_steer) / acc_base * 100}


def run_kmmlu(model, tok, df, shots, cfg, condition="base", max_tokens=24000, capture_layers=None,
              use_prefix_cache=False):
    """Letter-logit predictions for a KMMLU frame. If capture_layers is given, returns (frame, acts [L_sel, n, d]).

    Default: full k-shot prompts. The prefix-cache path is NOT valid with official AlphaSteer steering (the
    last-token Delta h is broadcast over the shared prefix too) and differs from the full path by bf16 noise
    (~1.3% prediction flips), so every condition uses the full path.
    """
    from prompts import kmmlu_prefix_suffix, kmmlu_prompt
    from scoring import score_texts, score_with_prefix
    out = df.copy()
    if not use_prefix_cache:
        texts = [kmmlu_prompt(tok, r, shots[r["subject"]], cfg, condition) for r in df.to_dict("records")]
        r = score_texts(model, tok, texts, "ABCD", max_tokens, capture_layers=capture_layers)
        for k in ("pred_idx", "margin", "p_letters"):
            out[k] = r[k]
        out["letter_logits"] = list(r["letter_logits"])
        return (out, r["acts"]) if capture_layers else out
    cols = {k: np.zeros(len(df), t) for k, t in
            [("pred_idx", np.int64), ("margin", np.float32), ("p_letters", np.float32)]}
    logits = np.zeros((len(df), 4), np.float32)
    acts = None
    for subj, idx in df.groupby("subject").indices.items():
        rows = df.iloc[idx].to_dict("records")
        pairs = [kmmlu_prefix_suffix(tok, r, shots[subj], cfg, condition) for r in rows]
        assert len({p for p, _ in pairs}) == 1
        r = score_with_prefix(model, tok, pairs[0][0], [s for _, s in pairs], "ABCD", max_tokens,
                              capture_layers=capture_layers)
        if capture_layers:
            if acts is None:
                acts = torch.zeros(len(capture_layers), len(df), r["acts"].shape[-1], dtype=torch.bfloat16)
            acts[:, torch.as_tensor(idx)] = r["acts"]
        for k in cols:
            cols[k][idx] = r[k]
        logits[idx] = r["letter_logits"]
    for k, v in cols.items():
        out[k] = v
    out["letter_logits"] = list(logits)
    return (out, acts) if capture_layers else out
