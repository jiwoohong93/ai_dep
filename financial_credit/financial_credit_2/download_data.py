"""Prepare the data.

    python download_data.py                 # check ./data, re-download missing files, install AIF360 raw files
    python download_data.py --only German Taiwan

The zip release bundles every file in ./data; the GitHub repository bundles all but the two largest
(Prosper, Home Credit). This script
  1. downloads any credit dataset file that is missing from ./data, and
  2. copies ./data/aif360/{adult,compas,german} into the installed aif360 package
     (aif360/data/raw/...), which `load_preproc_data_*` reads for dataset_name='adult'|'compas'|'german'.
"""
import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from algorithms.inprocessing.alincredit_function.loaders import DATA_DIR

URLS = {
    "German": ("german.data",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"),
    "Prosper": ("prosperLoanData.csv",
                "https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv"),
    "HomeCredit": ("homecredit_application_train.parquet",
                   "https://huggingface.co/datasets/mohameddhameem/home-credit-default-risk/resolve/main/"
                   "application_train_dated/train-00000-of-00001.parquet"),
    "HMEQ": ("hmeq.csv", "http://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv"),
    # 9,004-row extract of the LendingClub 2007-2011 accepted-loans file (LoanStats3a), hosted in this repository
    "LendingClub": ("lendingclub.csv", "https://raw.githubusercontent.com/jhoonjwa/ALINCredit/main/data/lendingclub.csv"),
}
TAIWAN_ZIP = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TAIWAN_FILE = "default of credit card clients.xls"
UCI_REPO = {"CreditApproval": (27, "creditapproval.csv"), "Australian": (143, "australian.csv")}
ALL = ["German", "LendingClub", "Taiwan", "Prosper", "HomeCredit", "HMEQ", "CreditApproval", "Australian"]

AIF360_FILES = {
    "adult": {f: f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/{f}"
              for f in ("adult.data", "adult.test", "adult.names")},
    "compas": {"compas-scores-two-years.csv":
               "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"},
    "german": {f: f"https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/{f}"
               for f in ("german.data", "german.doc")},
}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch(name):
    if name in URLS:
        fname, url = URLS[name]
        dst = os.path.join(DATA_DIR, fname)
        if not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(_get(url))
        return dst
    if name == "Taiwan":
        dst = os.path.join(DATA_DIR, TAIWAN_FILE)
        if not os.path.exists(dst):
            with zipfile.ZipFile(io.BytesIO(_get(TAIWAN_ZIP))) as z:
                member = next(n for n in z.namelist() if n.endswith(".xls"))
                with open(dst, "wb") as f:
                    f.write(z.read(member))
        return dst
    if name in UCI_REPO:
        uid, fname = UCI_REPO[name]
        dst = os.path.join(DATA_DIR, fname)
        if not os.path.exists(dst):
            import pandas as pd
            from ucimlrepo import fetch_ucirepo
            d = fetch_ucirepo(id=uid)
            pd.concat([d.data.features, d.data.targets], axis=1).to_csv(dst, index=False)
        return dst
    raise ValueError(name)


def install_aif360_raw():
    import aif360
    raw_dir = os.path.join(os.path.dirname(aif360.__file__), "data", "raw")
    for name, files in AIF360_FILES.items():
        for fname, url in files.items():
            src = os.path.join(DATA_DIR, "aif360", name, fname)
            if not os.path.exists(src):
                os.makedirs(os.path.dirname(src), exist_ok=True)
                with open(src, "wb") as f:
                    f.write(_get(url))
            dst = os.path.join(raw_dir, name, fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        print(f"[ok]   aif360/{name:<9} -> {os.path.join(raw_dir, name)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--skip-aif360", action="store_true", help="do not install the AIF360 benchmark raw files")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in args.only:
        try:
            print(f"[ok]   {name:<15} {fetch(name)}")
        except Exception as e:
            print(f"[miss] {name:<15} {e}")
    if not args.skip_aif360:
        install_aif360_raw()


if __name__ == "__main__":
    main()
