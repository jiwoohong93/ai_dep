"""Loaders for eight public credit-scoring datasets.

Each loader returns (R, kinds, y, sex):
    R     : pd.DataFrame of RAW (unfitted) feature columns, already under their final names
    kinds : {column: kind} telling the preprocessor how to fit each column on the train split
            "num"  -> z-score (mean/std fitted on train)
            "num_med" -> median-impute (median fitted on train), then z-score
            "cat"  -> ordinal codes (categories fitted on train; unseen -> NaN)
            "raw"  -> passed through unchanged (already numeric, no fitted statistics)
    y     : np.ndarray in {0, 1}, 1 = good (creditworthy / repaid)
    sex   : np.ndarray (1.0 = male, 0.0 = female) or None when unavailable

Sensitive attributes are NOT model inputs: sex (and attributes that encode it, e.g. German
A9 "personal status and sex") is returned separately and used only for fairness-aware
training weights and for evaluation.

Shared columns across datasets: "amount", "duration", "age". Every other attribute is kept
under a dataset-specific prefix (g_, lc_, tw_, pr_, hc_, hmeq_, ca_, aus_).

Raw files are read from DATA_DIR (default <package root>/data, override with the FAIRSCORE_DATA env var).
Run `python download_data.py` to fetch any missing file.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get(
    "FAIRSCORE_DATA",
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")), "data"))

# LendingClub: the first N terminal-status loans (file order) of the 2007-2011 export.
LC_MAX_ROWS = 9004
# Home Credit: random subsample size (seed 0).
HC_MAX_ROWS = 40000

DATA_FILES = {
    "German": "german.data",
    "LendingClub": "lendingclub.csv",
    "Taiwan": "default of credit card clients.xls",
    "Prosper": "prosperLoanData.csv",
    "HomeCredit": "homecredit_application_train.parquet",
    "HMEQ": "hmeq.csv",
    "CreditApproval": "creditapproval.csv",
    "Australian": "australian.csv",
}


def _path(name):
    return os.path.join(DATA_DIR, DATA_FILES[name])


def _num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def _pct(s):
    """'10.65%' -> 10.65 (no-op for already-numeric columns)."""
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


# --------------------------------------------------------------------------- German
def load_german():
    """Statlog German Credit (UCI id=144), 1,000 rows, 20 attributes."""
    cols = [f"A{i}" for i in range(1, 21)] + ["label"]
    g = pd.read_csv(_path("German"), sep=r"\s+", header=None, names=cols)
    y = (g["label"] == 1).astype(int).to_numpy()
    sex = np.where(g["A9"].isin({"A91", "A93", "A94"}), 1.0, 0.0)  # A9: personal status & sex
    R = pd.DataFrame({"amount": _num(g["A5"]), "duration": _num(g["A2"]), "age": _num(g["A13"])})
    kinds = {"amount": "num", "duration": "num", "age": "num"}
    for a in [c for c in cols[:-1] if c not in ("A5", "A2", "A13", "A9")]:  # A9 excluded (encodes sex)
        is_cat = g[a].astype(str).str.startswith("A").any()
        R["g_" + a] = g[a].astype(str) if is_cat else _num(g[a])
        kinds["g_" + a] = "cat" if is_cat else "num"
    return R, kinds, y, sex


# --------------------------------------------------------------------- LendingClub
def _read_lendingclub(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    skip = 0 if "loan_amnt" in first else 1  # raw LoanStats exports start with a notice line
    return pd.read_csv(path, skiprows=skip, low_memory=False)


def load_lendingclub():
    """LendingClub accepted loans 2007-2011. good = Fully Paid; bad = Charged Off / Default."""
    lc = _read_lendingclub(_path("LendingClub"))
    lc = lc[lc["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].head(LC_MAX_ROWS).reset_index(drop=True)
    y = (lc["loan_status"] == "Fully Paid").astype(int).to_numpy()
    R = pd.DataFrame({"amount": _num(lc["loan_amnt"]),
        "duration": _num(lc["term"].astype(str).str.extract(r"(\d+)")[0]),
        "lc_int": _pct(lc["int_rate"]), "lc_inc": _num(lc["annual_inc"]), "lc_dti": _num(lc["dti"]),
        "lc_emp": lc["emp_length"].astype(str), "lc_home": lc["home_ownership"].astype(str),
        "lc_revol": _pct(lc["revol_util"]), "lc_openacc": _num(lc["open_acc"]),
        "lc_pubrec": _num(lc["pub_rec"]), "lc_totacc": _num(lc["total_acc"])})
    kinds = {c: "num" for c in R.columns}
    kinds.update({"lc_emp": "cat", "lc_home": "cat"})
    return R, kinds, y, None


# -------------------------------------------------------------------------- Taiwan
def load_taiwan():
    """Default of Credit Card Clients, Taiwan (UCI id=350), 30,000 rows."""
    tw = pd.read_excel(_path("Taiwan"), header=1)
    y = (tw[tw.columns[-1]] == 0).astype(int).to_numpy()
    sex = (tw["SEX"] == 1).astype(float).to_numpy()
    R = pd.DataFrame({"amount": _num(tw["LIMIT_BAL"]), "age": _num(tw["AGE"]),
        "tw_edu": tw["EDUCATION"].astype(float), "tw_marr": tw["MARRIAGE"].astype(float)})
    kinds = {"amount": "num", "age": "num", "tw_edu": "raw", "tw_marr": "raw"}
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
              "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]:
        R["tw_" + c] = _num(tw[c]); kinds["tw_" + c] = "num"
    return R, kinds, y, sex


# ------------------------------------------------------------------------- Prosper
def load_prosper():
    """Prosper Loan Data. good = Completed; bad = Chargedoff / Defaulted. Non-terminal dropped."""
    df = pd.read_csv(_path("Prosper"), low_memory=False)
    df = df[df["LoanStatus"].isin(["Completed", "Chargedoff", "Defaulted"])].reset_index(drop=True)
    y = (df["LoanStatus"] == "Completed").astype(int).to_numpy()
    R = pd.DataFrame({
        "amount": _num(df["LoanOriginalAmount"]), "duration": _num(df["Term"]),
        "pr_rate": _num(df["BorrowerRate"]),
        "pr_listcat": df["ListingCategory (numeric)"].astype(float),
        "pr_homeowner": df["IsBorrowerHomeowner"].astype(float),
        "pr_emp": df["EmploymentStatus"].astype(str), "pr_incrange": df["IncomeRange"].astype(str),
        "pr_credlow": _num(df["CreditScoreRangeLower"]), "pr_dti": _num(df["DebtToIncomeRatio"]),
        "pr_inc": _num(df["StatedMonthlyIncome"]),
    })
    kinds = {c: "num" for c in R.columns}
    kinds.update({"pr_listcat": "raw", "pr_homeowner": "raw", "pr_emp": "cat", "pr_incrange": "cat"})
    return R, kinds, y, None


# ---------------------------------------------------------------------- Home Credit
def load_homecredit():
    """Home Credit Default Risk (application_train). good = TARGET 0. Sex from CODE_GENDER."""
    cols = ["TARGET", "CODE_GENDER", "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY",
            "AMT_INCOME_TOTAL", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_EMPLOYED"]
    hc = pd.read_parquet(_path("HomeCredit"), columns=cols)
    hc = hc[hc["CODE_GENDER"].isin(["M", "F"])]
    if len(hc) > HC_MAX_ROWS:
        hc = hc.sample(n=HC_MAX_ROWS, random_state=0)
    hc = hc.reset_index(drop=True)
    y = (hc["TARGET"] == 0).astype(int).to_numpy()
    sex = (hc["CODE_GENDER"] == "M").astype(float).to_numpy()

    emp = _num(hc["DAYS_EMPLOYED"])
    emp_anom = (emp == 365243).astype(float)  # sentinel for pensioner / unemployed
    R = pd.DataFrame({
        "amount": _num(hc["AMT_CREDIT"]), "age": -_num(hc["DAYS_BIRTH"]) / 365.0,
        "hc_annuity": _num(hc["AMT_ANNUITY"]), "hc_income": _num(hc["AMT_INCOME_TOTAL"]),
        "hc_housing": hc["NAME_HOUSING_TYPE"].astype(str), "hc_edu": hc["NAME_EDUCATION_TYPE"].astype(str),
        "hc_ext1": _num(hc["EXT_SOURCE_1"]), "hc_ext2": _num(hc["EXT_SOURCE_2"]), "hc_ext3": _num(hc["EXT_SOURCE_3"]),
        "hc_days_employed": emp.mask(emp == 365243), "hc_emp_anom": emp_anom,
    })
    kinds = {c: "num" for c in R.columns}
    kinds.update({"hc_housing": "cat", "hc_edu": "cat", "hc_emp_anom": "raw"})
    return R, kinds, y, sex


# ---------------------------------------------------------------------------- HMEQ
def load_hmeq():
    """HMEQ home-equity loans (Baesens et al., Credit Risk Analytics). good = BAD 0."""
    df = pd.read_csv(_path("HMEQ"))
    df.columns = [c.strip().upper() for c in df.columns]
    y = (df["BAD"].astype(int) == 0).astype(int).to_numpy()
    R = pd.DataFrame({"amount": _num(df["LOAN"])})
    kinds = {"amount": "num"}
    # numeric: missingness flag (informative in HMEQ) + train-median imputation + z-score
    for col in ["MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]:
        s = _num(df[col])
        if s.isna().any():
            R[f"hmeq_{col.lower()}_missing"] = s.isna().astype(float)
            kinds[f"hmeq_{col.lower()}_missing"] = "raw"
        R[f"hmeq_{col.lower()}"] = s
        kinds[f"hmeq_{col.lower()}"] = "num_med"
    for col in ["REASON", "JOB"]:
        R[f"hmeq_{col.lower()}"] = df[col].fillna("missing").astype(str)
        kinds[f"hmeq_{col.lower()}"] = "cat"
    return R, kinds, y, None


# ------------------------------------------------------------------ Credit Approval
def load_creditapproval():
    """Credit Approval / crx (UCI id=27), 690 rows. good = '+'."""
    ca = pd.read_csv(_path("CreditApproval"))
    y = (ca["A16"].astype(str) == "+").astype(int).to_numpy()
    R = pd.DataFrame({"amount": _num(ca["A15"]), "age": _num(ca["A2"])})
    kinds = {"amount": "num", "age": "num"}
    for a in ["A3", "A8", "A11", "A14"]:
        R["ca_" + a] = _num(ca[a]); kinds["ca_" + a] = "num"
    for a in ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]:
        R["ca_" + a] = ca[a].astype(str); kinds["ca_" + a] = "cat"
    return R, kinds, y, None


# ----------------------------------------------------------------------- Australian
def load_australian():
    """Statlog Australian Credit Approval (UCI id=143), 690 rows. good = class 1."""
    df = pd.read_csv(_path("Australian"))
    y = (df["A15"] == 1).astype(int).to_numpy()
    cont = ["A2", "A3", "A7", "A10", "A13", "A14"]
    R = pd.DataFrame({"amount": _num(df["A14"])})
    kinds = {"amount": "num"}
    for a in [f"A{i}" for i in range(1, 15)]:
        R["aus_" + a] = _num(df[a]) if a in cont else df[a].astype(float)
        kinds["aus_" + a] = "num" if a in cont else "raw"
    return R, kinds, y, None


# Order matters: datasets are stacked in this order for training.
LOADERS = {
    "German": load_german,
    "LendingClub": load_lendingclub,
    "Taiwan": load_taiwan,
    "Prosper": load_prosper,
    "HomeCredit": load_homecredit,
    "HMEQ": load_hmeq,
    "CreditApproval": load_creditapproval,
    "Australian": load_australian,
}
