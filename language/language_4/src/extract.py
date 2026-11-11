"""Resumable activation extraction to NAS shards.

Layout: <acts>/<run>/<name>/shard_00000.pt  -> {"acts": bf16 [L, n, d] (layer-major), "rows": int64 [n]}
        <acts>/<run>/<name>/shard_00000.parquet -> per-row scoring meta (preds, letter logits, margin)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scoring import score_texts


def disk_estimate_gb(n, n_layers, d):
    return n * n_layers * d * 2 / 1e9


def extract(model, tok, texts, meta, out_dir, letters, layers, shard_size=2048, max_tokens=24000, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    n_shards = (n + shard_size - 1) // shard_size
    log(f"[extract] {out_dir.name}: n={n} shards={n_shards} est_disk={disk_estimate_gb(n, len(layers), model.config.hidden_size):.1f}GB")
    t0, done = time.time(), 0
    for s in range(n_shards):
        f = out_dir / f"shard_{s:05d}.pt"
        if f.exists() and (out_dir / f"shard_{s:05d}.parquet").exists():
            continue
        lo, hi = s * shard_size, min(n, (s + 1) * shard_size)
        r = score_texts(model, tok, texts[lo:hi], letters, max_tokens, capture_layers=layers)
        m = meta.iloc[lo:hi].copy()
        m["row"] = np.arange(lo, hi)
        for k in ("pred_idx", "margin", "p_letters", "p_space_letters", "n_tokens"):
            m[k] = r[k]
        m["letter_logits"] = list(r["letter_logits"])
        tmp = f.with_suffix(".tmp")
        torch.save({"acts": r["acts"].contiguous(), "rows": torch.arange(lo, hi)}, tmp)
        tmp.rename(f)
        m.to_parquet(out_dir / f"shard_{s:05d}.parquet")
        done += hi - lo
        el = time.time() - t0
        log(f"[extract] {out_dir.name} shard {s + 1}/{n_shards} ({done / el:.1f} prompts/s, "
            f"eta {(n - hi) / max(done / el, 1e-9) / 60:.1f} min)")
    (out_dir / "DONE").write_text(str(n))


def load_meta(out_dir):
    fs = sorted(Path(out_dir).glob("shard_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_layer(out_dir, layer, rows=None, dtype=torch.float32):
    """Activations of one layer index (position in the saved `layers` list) -> [n, d]."""
    fs = sorted(Path(out_dir).glob("shard_*.pt"))
    parts = [torch.load(f, mmap=True)["acts"][layer].to(dtype) for f in fs]
    x = torch.cat(parts, 0)
    return x if rows is None else x[torch.as_tensor(rows)]


def load_layers(out_dir, layers, dtype=torch.float32):
    """-> [len(layers), n, d]"""
    fs = sorted(Path(out_dir).glob("shard_*.pt"))
    parts = []
    for f in fs:
        a = torch.load(f, mmap=True)["acts"]
        parts.append(a[list(layers)].to(dtype))
    return torch.cat(parts, 1)
