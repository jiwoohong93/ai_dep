"""Model, fairness metric, fairness-aware training weights and threshold selection.

The sensitive attribute is never a model input and never used at decision time:
  - training : group-and-label reweighing (Kamiran & Calders, 2012) on sex-labeled rows
  - decision : one global threshold shared by everyone
  - selection: the threshold is chosen on a validation split, subject to a DEO cap
"""
import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

# inputs are plain arrays; LightGBM's auto-generated feature names trigger this harmless warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class Ensemble:
    """NaN-native soft-vote of HistGradientBoosting + LightGBM."""

    def __init__(self, random_state=0, n_jobs=None):
        self.random_state = random_state
        self.n_jobs = n_jobs  # LightGBM threads (None = all cores)

    def fit(self, X, y, sample_weight=None):
        self.h = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
            random_state=self.random_state).fit(X, y, sample_weight=sample_weight)
        self.l = LGBMClassifier(n_estimators=450, learning_rate=0.03, num_leaves=31, min_child_samples=25,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=self.random_state, n_jobs=self.n_jobs or -1, verbose=-1).fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return (self.h.predict_proba(X) + self.l.predict_proba(X)) / 2


def deo_x100(y_true, y_pred, sex):
    """Difference of Equal Opportunity x100 = |TPR_male - TPR_female| * 100."""
    tpr = {}
    for g in (1.0, 0.0):
        m = (sex == g) & (y_true == 1)
        if m.sum() == 0:
            return np.nan
        tpr[g] = y_pred[m].mean()
    return abs(tpr[1.0] - tpr[0.0]) * 100


def reweigh(y, sex, dataset_ids):
    """Kamiran-Calders weights w(g, y) = P(g) P(y) / P(g, y), computed per dataset.

    Rows without a sex label (sex is NaN) keep weight 1.
    """
    w = np.ones(len(y), dtype=float)
    for d in np.unique(dataset_ids):
        m = (dataset_ids == d) & ~np.isnan(sex)
        if not m.any():
            continue
        ys, ss = y[m], sex[m]
        wd = np.ones(m.sum())
        for g in (0.0, 1.0):
            for c in (0, 1):
                cell = (ss == g) & (ys == c)
                if cell.any():
                    wd[cell] = (ss == g).mean() * (ys == c).mean() / cell.mean()
        w[m] = wd
    return w


GRID = np.arange(0.20, 0.80, 0.005)


def select_threshold(p, y, sex=None, cap=None):
    """Global threshold maximizing accuracy; with `cap`, subject to DEO(sex-labeled rows) <= cap.

    Falls back to the threshold with the smallest DEO if no threshold satisfies the cap.
    """
    acc = np.array([((p >= t).astype(int) == y).mean() for t in GRID])
    if cap is None:
        return float(GRID[acc.argmax()])
    m = ~np.isnan(sex)
    deo = np.array([deo_x100(y[m], (p[m] >= t).astype(int), sex[m]) for t in GRID])
    ok = deo <= cap
    if ok.any():
        return float(GRID[np.where(ok, acc, -1).argmax()])
    return float(GRID[np.nanargmin(deo)])
