"""KoBBQ loading, normalization (3-way roles) and template-level splits."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("unknown", "stereo", "anti")


def load_kobbq(kobbq_dir, unknown_text="알 수 없음"):
    """Normalize KoBBQ_all_samples.tsv.

    sample_id = {category}-{templateNNN}{version}-{substitution}-{amb|dis}-{bsd|cnt}
    version a/c = counter-biased context, b/d = biased context (official 5_evaluation.py).
    stereo = `biased_answer` (answer conforming to the stereotype for *that* question direction),
    unknown = the single "알 수 없음" choice, anti = the remaining choice.
    """
    df = pd.read_csv(Path(kobbq_dir) / "KoBBQ_all_samples.tsv", sep="\t")
    df["choices"] = df["choices"].map(ast.literal_eval)
    parts = df["sample_id"].str.split("-", expand=True)
    out = pd.DataFrame({
        "sample_id": df["sample_id"],
        "category": parts[0],
        "template_id": parts[0] + "-" + parts[1].str[:-1],
        "version": parts[1].str[-1],
        "context_type": parts[3].map({"amb": "ambig", "dis": "disambig"}),
        "question_polarity": parts[4].map({"bsd": "biased", "cnt": "counter_biased"}),
        "label_annotation": df["label_annotation"],
        "context": df["context"],
        "question": df["question"],
    })
    out["context_bias"] = np.where(out["version"].isin(["b", "d"]), "biased", "counter_biased")

    unk, stereo, anti = [], [], []
    for ch, b in zip(df["choices"], df["biased_answer"]):
        assert len(ch) == 3 and len(set(ch)) == 3, ch
        u = [c for c in ch if c == unknown_text]
        assert len(u) == 1, ch
        assert b in ch and b != unknown_text, (ch, b)
        a = [c for c in ch if c not in (unknown_text, b)]
        assert len(a) == 1
        unk.append(u[0]); stereo.append(b); anti.append(a[0])
    out["unknown_text"], out["stereo_text"], out["anti_text"] = unk, stereo, anti

    gold = []
    for ans, u, s, a in zip(df["answer"], unk, stereo, anti):
        gold.append({u: "unknown", s: "stereo", a: "anti"}[ans])
    out["gold_role"] = gold
    # sanity: ambig gold is always unknown; disamb gold is stereo iff the context is biased
    amb = out.context_type == "ambig"
    assert (out.loc[amb, "gold_role"] == "unknown").all()
    dis = out[~amb]
    assert ((dis.gold_role == "stereo") == (dis.context_bias == "biased")).all()
    return out


def template_split(df, ratios=(0.6, 0.2, 0.2), seed=42):
    """Category-stratified template split: every sample of a template goes to one split."""
    rng = np.random.RandomState(seed)
    mapping = {}
    for cat in sorted(df.category.unique()):
        tids = sorted(df.loc[df.category == cat, "template_id"].unique())
        rng.shuffle(tids)
        n = len(tids)
        n_val = max(1, int(round(n * ratios[1])))
        n_test = max(1, int(round(n * ratios[2])))
        n_train = n - n_val - n_test
        assert n_train >= 1
        for t in tids[:n_train]:
            mapping[t] = "train"
        for t in tids[n_train:n_train + n_val]:
            mapping[t] = "val"
        for t in tids[n_train + n_val:]:
            mapping[t] = "test"
    cats = df[["template_id", "category"]].drop_duplicates()
    cats["split"] = cats.template_id.map(mapping)
    for cat, g in cats.groupby("category"):
        assert set(g.split) == {"train", "val", "test"}, cat
    return mapping


def template_kfold(df, n_folds=5, seed=42):
    """Category-stratified GroupKFold over templates: template_id -> fold index."""
    rng = np.random.RandomState(seed)
    mapping = {}
    for cat in sorted(df.category.unique()):
        tids = sorted(df.loc[df.category == cat, "template_id"].unique())
        rng.shuffle(tids)
        for i, t in enumerate(tids):
            mapping[t] = i % n_folds
    return mapping
