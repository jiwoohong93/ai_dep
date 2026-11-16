"""
test_edu_gbm.py — SOUND grader for the coarse-target 최종학력 classifier.

Computes the three assignment metrics with test.py's EXACT definitions, but on a
leakage-free evaluation (no oversample-before-split, the flaw that inflated the
original numbers):

  1. 분류 정확도 (%)          accuracy_score(fair)                    must be > 85
  2. 공정성 (DEO)             abs(TPR_unprot − TPR_prot) on '고졸'     must be < 0.07
  3. 정책 반영 평균 하락률 (%)  |fair_acc − unfair_acc|                 must be < 5

Two complementary views of every metric:

  * POINT estimate  — train_edu_gbm.py's fixed honest split (seed 42, 10% held
    out), reusing the SAVED fair/unfair pickles.  Validates the persisted models
    exactly as an inference client would load them.

  * ROBUST estimate — full out-of-fold predictions (StratifiedKFold k=5, feature
    state re-fit on each fold's TRAIN only).  Every row gets a prediction from a
    model that never saw it, so it stays leakage-free AND maximizes the
    protected-고졸 support — the fix for the fact that a single 10% split holds
    only ~1 protected 고졸 person, making its DEO a coin flip.  Per-fold DEO
    spread is printed so the fairness result is never a single noisy number.

The DEO function is copied VERBATIM from test.py (absolute formula) so the number
is definitionally identical to the grader; the protected attribute is the same
women's-school proxy (학교명 contains '여자' → +1, else −1).

Run:
    conda activate ai_dep
    python train_edu_gbm.py --fairness none          # unfair baseline pkl
    python train_edu_gbm.py                          # fair pkl (default config)
    python test_edu_gbm.py
"""
import argparse
import logging
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

import train_edu_gbm as T

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

POS = "고졸"                       # coarse high-school class (coarsen() output)
ACC_MIN, DEO_MAX, DEG_MAX = 0.85, 0.07, 0.05
# Minimum protected-positive (여자-school 고졸) support for a DEO number to be a
# real measurement. Below this the group has too few members to estimate a TPR —
# TPR+ collapses to the denom==0 → 0.0 fallback, so any "DEO" is an artifact, not
# a fairness result. The whole point of the OOF view is to pool enough support.
MIN_PROT_POS = 3


# --- DEO copied VERBATIM from test.py (absolute formula) for metric fidelity ---
def measure_deo(y_true, y_pred, protected_group, positive_label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    protected_group = np.array(protected_group)

    mask_neg = (protected_group == -1)
    mask_pos = (protected_group == 1)

    denom_neg = np.sum(y_true[mask_neg] == positive_label)
    tpr_neg = 0.0 if denom_neg == 0 else np.sum(
        (y_true[mask_neg] == positive_label) & (y_pred[mask_neg] == positive_label)
    ) / denom_neg

    denom_pos = np.sum(y_true[mask_pos] == positive_label)
    tpr_pos = 0.0 if denom_pos == 0 else np.sum(
        (y_true[mask_pos] == positive_label) & (y_pred[mask_pos] == positive_label)
    ) / denom_pos

    return abs(tpr_neg - tpr_pos), tpr_neg, tpr_pos


def make_weight_fn(fairness, floor, strength):
    """Return a f(y_train) -> sample weights closure for a fairness config."""
    if fairness == "minority":
        return lambda ytr: T.compute_minority_weights(ytr, floor)
    if fairness == "reweight":
        return lambda ytr: T.compute_sample_weights(ytr, strength)
    return lambda ytr: np.ones(len(ytr))          # 'none' (unfair)


def load_prot(df):
    return df[T.PROTECTED_COL].astype(str).apply(lambda x: 1 if "여자" in x else -1).values


# --------------------------------------------------------------------------- #
# POINT estimate — reuse the saved pickles on the fixed seed-42 honest split   #
# --------------------------------------------------------------------------- #
def point_estimate(df, y, y_fine, prot, fair_pkl, unfair_pkl, test_size, seed):
    idx = np.arange(len(df))
    _, ite = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y_fine)
    test_df = df.iloc[ite]
    yte = y.iloc[ite].values
    prot_te = prot[ite]

    def predict_saved(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        Xte = T.transform_features(test_df, d["feature_state"])[d["feature_columns"]]
        return d["model"].predict(Xte)

    fair_pred = predict_saved(fair_pkl)
    unfair_pred = predict_saved(unfair_pkl)
    fair_acc = accuracy_score(yte, fair_pred)
    unfair_acc = accuracy_score(yte, unfair_pred)
    deo, tn, tp = measure_deo(yte, fair_pred, prot_te, POS)
    n_prot_hs = int(((prot_te == 1) & (yte == POS)).sum())
    return dict(fair_acc=fair_acc, unfair_acc=unfair_acc, deo=deo, tpr_neg=tn,
                tpr_pos=tp, n_prot_hs=n_prot_hs, n_test=len(ite))


# --------------------------------------------------------------------------- #
# ROBUST estimate — leakage-free full out-of-fold predictions                  #
# --------------------------------------------------------------------------- #
def oof_predict(df, y, folds, cat_cols, weight_fn):
    oof = pd.Series(index=np.arange(len(df)), dtype=object)
    per_fold_deo, prot = [], load_prot(df)
    for tr, te in folds:
        train_df, test_df = df.iloc[tr], df.iloc[te]
        ytr = y.iloc[tr]
        state = T.fit_feature_state(train_df)
        feat = T.feature_columns(state)
        Xtr = T.transform_features(train_df, state)[feat]
        Xte = T.transform_features(test_df, state)[feat]
        model = T.build_model("lgbm")
        model = T.fit_model(model, "lgbm", Xtr, ytr, cat_cols, weight_fn(ytr))
        pred = model.predict(Xte)
        oof.iloc[te] = pred
        d, _, _ = measure_deo(y.iloc[te].values, pred, prot[te], POS)
        per_fold_deo.append(d)
    return oof.values, per_fold_deo


def robust_estimate(df, y, y_fine, prot, fair_cfg, seed):
    cat_cols = T._categorical_columns()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                 .split(np.arange(len(df)), y_fine))
    fair_pred, pf = oof_predict(df, y, folds, cat_cols, make_weight_fn(**fair_cfg))
    unfair_pred, _ = oof_predict(df, y, folds, cat_cols,
                                 make_weight_fn("none", None, None))
    fair_acc = accuracy_score(y.values, fair_pred)
    unfair_acc = accuracy_score(y.values, unfair_pred)
    deo, tn, tp = measure_deo(y.values, fair_pred, prot, POS)
    n_prot_hs = int(((prot == 1) & (y.values == POS)).sum())
    return dict(fair_acc=fair_acc, unfair_acc=unfair_acc, deo=deo, tpr_neg=tn,
                tpr_pos=tp, per_fold_deo=pf, n_prot_hs=n_prot_hs)


# --------------------------------------------------------------------------- #
def verdict(name, acc, deo, deg, deo_measurable=True):
    """Print the three-threshold verdict.

    If ``deo_measurable`` is False the protected-positive support was too small
    for DEO to be a real measurement, so DEO is reported as N/A and EXCLUDED from
    the pass decision (an unmeasurable quantity is neither PASS nor FAIL).
    """
    a = "PASS" if acc > ACC_MIN else "FAIL"
    g = "PASS" if deg < DEG_MAX else "FAIL"
    print(f"\n=== {name} — thresholds ===")
    print(f"  분류 정확도 (%)         : {acc*100:7.3f}   (> 85)   [{a}]")
    if deo_measurable:
        d = "PASS" if deo < DEO_MAX else "FAIL"
        print(f"  공정성 (DEO)            : {deo:7.4f}   (< 0.07) [{d}]")
    else:
        d = "N/A"
        print(f"  공정성 (DEO)            : {deo:7.4f}   (< 0.07) [N/A — support too low]")
    print(f"  정책 반영 평균 하락률 (%) : {deg*100:7.3f}   (< 5)    [{g}]")
    passed = (a == "PASS") and (g == "PASS") and (d in ("PASS", "N/A"))
    return passed, d


def parse_args():
    p = argparse.ArgumentParser(description="Sound grader for coarse 최종학력 classifier.")
    p.add_argument("--data", default="data_stage1.csv")
    p.add_argument("--fair-fairness", choices=["minority", "reweight"], default="minority",
                   help="Fairness lever of the FAIR model (must match its training).")
    p.add_argument("--fair-floor", type=float, default=250, help="minority floor (shipped=250, num_leaves=63).")
    p.add_argument("--fair-strength", type=float, default=0.25, help="reweight strength.")
    p.add_argument("--fair-pkl", default=None, help="default: models/trained_edu_<fair-fairness>.pkl")
    p.add_argument("--unfair-pkl", default="models/trained_edu_none.pkl")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-point", action="store_true",
                   help="skip the saved-pickle point estimate (OOF only).")
    return p.parse_args()


def main():
    args = parse_args()
    fair_pkl = args.fair_pkl or f"models/trained_edu_{args.fair_fairness}.pkl"
    fair_cfg = dict(fairness=args.fair_fairness, floor=args.fair_floor,
                    strength=args.fair_strength)

    df = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)
    y_fine = df[T.TARGET_COL].astype(str)
    y = y_fine.apply(T.coarsen)
    prot = load_prot(df)

    cfg_str = (f"minority floor={args.fair_floor:.0f}" if args.fair_fairness == "minority"
               else f"reweight s={args.fair_strength}")
    print(f"\nFAIR model = {cfg_str}   |   UNFAIR = none")
    print(f"data: {len(df)} rows | 고졸 total={int((y==POS).sum())} "
          f"| protected(여자) 고졸={int(((prot==1)&(y.values==POS)).sum())}")

    all_pass = True

    # ---- POINT estimate from saved pickles (diagnostic; not the DEO authority) ----
    # A 10% split holds only ~0-1 protected 고졸 people, so its DEO is unmeasurable
    # (TPR+ falls back to 0.0). We still report point accuracy + degradation — those
    # ARE meaningful on 1000 rows — but the point DEO is flagged N/A and the OOF view
    # below is the authoritative fairness number.
    if not args.skip_point and os.path.exists(fair_pkl) and os.path.exists(args.unfair_pkl):
        logging.info("Point estimate on the fixed seed-%d split (saved pickles)…", args.seed)
        r = point_estimate(df, y, y_fine, prot, fair_pkl, args.unfair_pkl,
                           args.test_size, args.seed)
        measurable = r["n_prot_hs"] >= MIN_PROT_POS
        print(f"\n--- POINT estimate (fixed {args.test_size:.0%} split, "
              f"n_test={r['n_test']}, protected 고졸 in test={r['n_prot_hs']}) ---")
        print(f"  fair acc={r['fair_acc']:.4f}  unfair acc={r['unfair_acc']:.4f}  "
              f"TPR-={r['tpr_neg']:.3f} TPR+={r['tpr_pos']:.3f}")
        ppass, _ = verdict("POINT", r["fair_acc"], r["deo"],
                           abs(r["fair_acc"] - r["unfair_acc"]), deo_measurable=measurable)
        all_pass &= ppass
        if not measurable:
            print(f"  (point-split DEO excluded: only {r['n_prot_hs']} protected 고졸 "
                  f"person(s) < {MIN_PROT_POS} — equality-of-opportunity is not estimable "
                  f"on one split; the ROBUST view pools all protected examples)")
    else:
        logging.warning("Skipping point estimate (pickles missing or --skip-point).")

    # ---- ROBUST estimate via OOF (the authoritative fairness verdict) ----
    logging.info("Robust estimate via 5-fold OOF (leakage-free, per-fold FE fit)…")
    r = robust_estimate(df, y, y_fine, prot, fair_cfg, args.seed)
    pf = ",".join(f"{x:.3f}" for x in r["per_fold_deo"])
    print(f"\n--- ROBUST estimate (full out-of-fold, protected 고졸 support={r['n_prot_hs']}) ---")
    print(f"  fair acc={r['fair_acc']:.4f}  unfair acc={r['unfair_acc']:.4f}  "
          f"TPR-={r['tpr_neg']:.3f} TPR+={r['tpr_pos']:.3f}")
    print(f"  per-fold abs-DEO = [{pf}]  (resolution ≈ 1/{r['n_prot_hs']}="
          f"{1.0/max(r['n_prot_hs'],1):.3f} per protected person)")
    rpass, _ = verdict("ROBUST (OOF)", r["fair_acc"], r["deo"],
                       abs(r["fair_acc"] - r["unfair_acc"]))
    all_pass &= rpass

    print(f"\n{'='*60}\nOVERALL: {'ALL THRESHOLDS PASS' if all_pass else 'SOME THRESHOLDS FAIL'}\n{'='*60}")


if __name__ == "__main__":
    main()
