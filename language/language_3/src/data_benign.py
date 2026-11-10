"""Korean benign prompts for H_b (null-space estimation). KMMLU / KoBBQ are excluded."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HANGUL = re.compile(r"[가-힣]")


def norm_text(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\W_]+", "", s)


def hangul_ratio(s):
    chars = [c for c in s if not c.isspace()]
    return len(HANGUL.findall(s)) / max(1, len(chars))


def _join(instr, inp):
    instr, inp = str(instr or "").strip(), str(inp or "").strip()
    return instr if not inp else f"{instr}\n\n{inp}"


def load_sources(raw):
    raw = Path(raw)
    src = {}
    rows = [json.loads(l) for l in open(raw / "nlpai-lab__kullm-v2/kullm-v2.jsonl")]
    src["kullm_v2"] = pd.DataFrame({"text": [_join(r["instruction"], r.get("input")) for r in rows],
                                    "sub": [r["id"].split("_")[0] for r in rows]})
    d = pd.read_parquet(next((raw / "kyujinpy__KOpen-platypus/data").glob("*.parquet")))
    src["kopen_platypus"] = pd.DataFrame({"text": [_join(a, b) for a, b in zip(d.instruction, d.input)],
                                          "sub": d.data_source})
    d = pd.read_parquet(raw / "kuotient__gsm8k-ko/data/train-00000-of-00001.parquet")
    src["gsm8k_ko"] = pd.DataFrame({"text": d.question.str.strip(), "sub": "gsm8k_train"})
    rows = [json.loads(l) for l in open(raw / "CarrotAI__ko-code-alpaca-QA/ko-code-alpaca-QA.jsonl")]
    src["ko_code_alpaca_qa"] = pd.DataFrame({"text": [_join(r["instruction"], r.get("input")) for r in rows],
                                             "sub": "ko_code_alpaca_qa"})
    return src


def build_benign(cfg, exclude_texts, log=print):
    """Sample per-source quotas, filter, dedup (exact + normalized) against KMMLU/KoBBQ, split 90:10."""
    bcfg, seed = cfg["benign"], cfg["seed"]
    rng = np.random.RandomState(seed)
    excl = {norm_text(t) for t in exclude_texts}
    excl.discard("")
    seen, keep, stats = set(), [], {}
    for name, df in load_sources(cfg["paths"]["raw"]).items():
        n0 = len(df)
        df = df[df.text.str.len().between(5, bcfg["max_chars"])]
        df = df[df.text.map(hangul_ratio) >= bcfg["min_hangul_ratio"]]
        n_filt = len(df)
        df = df.iloc[rng.permutation(len(df))]
        quota, taken, dup, overlap = bcfg["sources"][name], 0, 0, 0
        for t, sub in zip(df.text, df["sub"]):
            if taken >= quota:
                break
            k = norm_text(t)
            if t in exclude_texts or k in excl:  # exact / normalized-string overlap
                overlap += 1
                continue
            if t in seen or k in seen:
                dup += 1
                continue
            seen.add(t); seen.add(k)
            keep.append({"text": t, "source": name, "sub": sub})
            taken += 1
        stats[name] = dict(raw=n0, after_filter=n_filt, taken=taken, dup_removed=dup, overlap_removed=overlap)
        log(f"benign {name}: {stats[name]}")
    out = pd.DataFrame(keep)
    perm = rng.permutation(len(out))
    n_val = int(round(len(out) * bcfg["val_frac"]))
    out["split"] = "train"
    out.loc[out.index[perm[:n_val]], "split"] = "val"
    out["benign_id"] = np.arange(len(out))
    return out, stats
