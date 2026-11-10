"""Run ALINCredit over several seeds and save every result.

    python evaluate.py                                  # credit_union, seeds 1..8
    python evaluate.py --seeds 3 --jobs 3
    python evaluate.py --dataset german --seeds 1

Writes results/<dataset>.json: mean / sd summary, every per-seed result (both metric
OrderedDicts from metrics.common_utils plus the per-dataset summary), package versions and
SHA-256 hashes of the data files used.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "aif360", "pyarrow", "xlrd"]
KEYS = ["accuracy", "Balanced accuracy", "equal_opportunity_difference", "average_odds_difference",
        "statistical_parity_difference", "disparate_impact"]


def run_one(dataset, seed, deo_cap, raw, threads):
    warnings.filterwarnings("ignore", category=FutureWarning)
    from threadpoolctl import threadpool_limits
    from algorithms.inprocessing.ALINCredit import ALINCredit
    with threadpool_limits(limits=threads):
        model = ALINCredit(dataset_name=dataset, deo_cap=deo_cap, seed=seed, n_jobs=threads, raw=raw)
        metrics_orig, metrics_transf = model.run()
    clean = lambda m: {k: float(v) for k, v in m.items()}
    rec = {"seed": seed, "metrics_orig": clean(metrics_orig), "metrics_transf": clean(metrics_transf), **model.summary}
    print(f"seed {seed}: macro acc {rec['macro_acc']:.2f}  DEO {rec['deo_x100']:.2f}  "
          f"drop {rec['policy_drop']:.2f}  (LR acc {100 * rec['metrics_orig']['accuracy']:.2f})", flush=True)
    return rec


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(dataset, names):
    from algorithms.inprocessing.alincredit_function import DATA_FILES, DATA_DIR
    files = {}
    if dataset == "credit_union":
        for k in names:
            p = os.path.join(DATA_DIR, DATA_FILES[k])
            files[k] = {"file": DATA_FILES[k], "sha256": sha256(p), "bytes": os.path.getsize(p)}
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    return {"python": platform.python_version(), "packages": versions, "data_files": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="credit_union", choices=["credit_union", "german", "adult", "compas"])
    ap.add_argument("--seeds", type=int, default=8, help="runs seeds 1..N")
    ap.add_argument("--deo-cap", type=float, default=4.0)
    ap.add_argument("--jobs", type=int, default=1, help="seeds run in parallel (CPU threads are split between them)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw, names = None, [args.dataset]
    if args.dataset == "credit_union":
        from algorithms.inprocessing.alincredit_function import load_raw
        raw = load_raw()
        names = list(raw[0])
        print(f"datasets ({len(names)}): {', '.join(names)} | union features: {len(raw[1])}")

    seeds = list(range(1, args.seeds + 1))
    threads = max(1, (os.cpu_count() or 1) // args.jobs)
    if args.jobs == 1:
        recs = [run_one(args.dataset, s, args.deo_cap, raw, threads) for s in seeds]
    else:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(run_one)(args.dataset, s, args.deo_cap, raw, threads) for s in seeds)

    mean = lambda xs: round(float(np.mean(xs)), 2)
    std = lambda xs: round(float(np.std(xs)), 2)
    summary = {
        "macro_acc": mean([r["macro_acc"] for r in recs]), "macro_acc_std": std([r["macro_acc"] for r in recs]),
        "pooled_acc": mean([r["pooled_acc"] for r in recs]),
        "deo_x100": mean([r["deo_x100"] for r in recs]), "deo_x100_std": std([r["deo_x100"] for r in recs]),
        "policy_drop": mean([r["policy_drop"] for r in recs]),
        "per_dataset_acc": {k: mean([r["per_dataset_acc"][k] for r in recs]) for k in names},
        "per_dataset_deo": {k: mean([r["per_dataset_deo"][k] for r in recs]) for k in recs[0]["per_dataset_deo"]},
        "metrics_orig": {k: round(float(np.mean([r["metrics_orig"][k] for r in recs])), 4) for k in KEYS},
        "metrics_transf": {k: round(float(np.mean([r["metrics_transf"][k] for r in recs])), 4) for k in KEYS},
    }
    out = {"dataset": args.dataset, "seeds": seeds, "deo_cap": args.deo_cap, "summary": summary,
           "runs": recs, "environment": environment(args.dataset, names)}

    print("=" * 66)
    print(f"{'dataset':<16}{'test acc %':>14}{'DEO x100':>12}")
    for k, v in summary["per_dataset_acc"].items():
        d = summary["per_dataset_deo"].get(k)
        print(f"{k:<16}{v:>14.2f}{'' if d is None else f'{d:>12.2f}'}")
    print("-" * 66)
    print(f"{'':<34}{'LR baseline':>14}{'proposed':>12}")
    for k in KEYS:
        print(f"{k:<34}{summary['metrics_orig'][k]:>14.4f}{summary['metrics_transf'][k]:>12.4f}")
    print("-" * 66)
    print(f"Accuracy (macro, per-dataset mean)  {summary['macro_acc']:.2f} % (sd {summary['macro_acc_std']:.2f})")
    print(f"DEO x100 = |equal_opportunity_difference| x 100   {summary['deo_x100']:.2f} (sd {summary['deo_x100_std']:.2f})")
    print(f"Policy drop %                       {summary['policy_drop']:.2f}")

    path = args.out or os.path.join(ROOT, "results", f"{args.dataset}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
