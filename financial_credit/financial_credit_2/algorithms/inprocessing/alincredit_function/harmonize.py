"""Train-split-fitted preprocessing and the union feature space.

Each dataset keeps all of its own features; features it does not have are NaN.
The tree models used downstream (HistGradientBoosting, LightGBM) handle NaN natively,
so no imputation across datasets is needed.
"""
import numpy as np
import pandas as pd

from .loaders import LOADERS


class Preprocessor:
    """Per-dataset preprocessing whose statistics are fitted on the train split only."""

    def __init__(self, kinds):
        self.kinds = kinds

    def fit(self, R):
        self.stats = {}
        for c, k in self.kinds.items():
            s = R[c]
            if k == "num":
                self.stats[c] = (s.mean(), s.std())
            elif k == "num_med":
                med = s.median()
                f = s.fillna(med)
                self.stats[c] = (med, f.mean(), f.std())
            elif k == "cat":
                self.stats[c] = {v: i for i, v in enumerate(sorted(s.dropna().unique()))}
        return self

    def transform(self, R):
        out = {}
        for c, k in self.kinds.items():
            s = R[c]
            if k == "num":
                mu, sd = self.stats[c]
                out[c] = (s - mu) / (sd + 1e-9) if np.isfinite(sd) else s * np.nan
            elif k == "num_med":
                med, mu, sd = self.stats[c]
                out[c] = (s.fillna(med) - mu) / (sd + 1e-9)
            elif k == "cat":
                out[c] = s.map(self.stats[c]).astype(float)  # unseen category -> NaN
            else:
                out[c] = s.astype(float)
        return pd.DataFrame(out, index=R.index)


def load_raw(names=None, verbose=True):
    """Return ({name: (R, kinds, y, sex_or_None)}, union_columns).

    names: optional subset of LOADERS keys. Datasets whose raw files are missing are skipped.
    """
    ds = {}
    for name, fn in LOADERS.items():
        if names is not None and name not in names:
            continue
        try:
            ds[name] = fn()
        except Exception as e:
            if verbose:
                print(f"[skip {name}] {type(e).__name__}: {str(e)[:120]}")
    if not ds:
        raise RuntimeError("No dataset could be loaded. Run `python download_data.py` first.")
    allcols = sorted(set().union(*[set(d[1]) for d in ds.values()]))
    return ds, allcols
