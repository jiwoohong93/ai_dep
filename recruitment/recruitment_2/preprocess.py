"""
preprocess.py — Stage 1 (Data Preprocessing) + Stage 2 (Feature Selection)
for the Scout recruitment dataset, sourced directly from the raw Excel DB.

This is a standalone reproduction of the first two pipeline stages described in
METHOD.md (§4, Stages 1-2) and implemented inline in train_fair.py /
train_unfair.py (Steps 1-2 and Steps 4/6). Instead of reading the intermediate
`data.csv`, it reads the raw source:

    scout_data/스카우트_이력사항DB_20221222.xlsx   (sheet '통합정보', 59 cols, 370,860 rows)

The constants and per-stage logic are kept identical to train_fair.py so the
output matches what the training scripts compute internally. The existing
train_*.py / test.py are left untouched.

Stage 1 — Data Preprocessing  (mirrors train_fair.py Steps 1-2)
    1. Load the Excel, strip + NFC-normalize column names.
    2. Subsample: df.sample(frac=SAMPLING_FRACTION, random_state=42)
       (0.02696435312 * 370,860 ≈ 10,001 rows — the ~10k working set per the deck).
    3. Drop rows whose target 최종학력 is blank; cast the label to str.
    4. Drop rare classes: keep only labels with count >= RARE_CLASS_THRESHOLD (5).

Stage 2 — Feature Selection  (mirrors train_fair.py Step 4 + the feature-clean
                              half of Step 6; ONE-HOT ENCODING IS NOT DONE HERE)
    1. De-duplicate SELECTED_FEATURES via list(set(...)) -> 10 unique columns,
       then intersect with the columns actually present.
    2. Keep those features + the target 최종학력 + the protected column 학교명
       (학교명 is retained so the downstream fairness / oversampling stages can
       still infer gender from women's-school names).
    3. On the feature columns: fillna("missing") then cast every value to str.

Outputs (CSV, utf-8-sig so Korean opens cleanly in Excel):
    data_stage1.csv        — post Stage 1, all 59 columns (for inspection)
    data_preprocessed.csv  — post Stage 2, 12 columns (target + 10 features + 학교명)

Caveats
    * Exact bit-reproduction of the report's data.csv sample is NOT guaranteed:
      df.sample reproduces the training split only if the Excel row order/count
      equals the original data.csv. Faithful stage *logic* is the goal, not
      reproducing 82.8% to the digit.
    * Numeric-looking fields (희망연봉, 최종연봉, 전체경력) are treated as strings,
      matching the training pipeline (they are one-hot encoded downstream).

Run:
    conda run -n ai_dep python preprocess.py
"""

import os
import logging
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (kept identical to train_fair.py for fidelity)
# ---------------------------------------------------------------------------
RAW_XLSX = "scout_data/스카우트_이력사항DB_20221222.xlsx"
SHEET_NAME = "통합정보"

TARGET_COL = "최종학력"
PROTECTED_COL = "학교명"

SAMPLING_FRACTION = 0.02696435312
RARE_CLASS_THRESHOLD = 5
RANDOM_STATE = 42

# Same list (including the intentional duplicate 희망(경력)업종1) as train_fair.py.
# list(set(...)) de-duplicates it down to 10 unique feature columns.
SELECTED_FEATURES = [
    "전체경력",
    "희망연봉",
    "근무형태",
    "희망(경력)직종1",
    "희망(경력)업종1",
    "근무지역",
    "희망(경력)세부직종1",
    "희망(경력)직종2",
    "직무태그",
    "희망(경력)업종1",
    "최종연봉",
]

STAGE1_CSV = "data_stage1.csv"
OUTPUT_CSV = "data_preprocessed.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _nfc(text):
    """NFC-normalize a header string so Korean names with parentheses/digits
    match the constant strings regardless of NFC/NFD storage."""
    return unicodedata.normalize("NFC", str(text).strip())


def load_raw_excel(path=RAW_XLSX, sheet_name=SHEET_NAME):
    """Load the raw Excel DB and normalize its column names.

    Note: this reads all ~370k rows of a ~165 MB workbook via openpyxl, which is
    slow and memory-heavy (a one-time cost). We must load the full frame before
    sampling so df.sample(frac=...) matches the training scripts' semantics.
    """
    if not os.path.exists(path):
        msg = f"Could not find raw Excel file: {path}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    logging.info("Reading Excel '%s' (sheet '%s') — this can take a few minutes...",
                 path, sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    # Match train_fair.py's df.columns.str.strip(), plus NFC normalization.
    df.columns = [_nfc(c) for c in df.columns]
    logging.info("Loaded raw DataFrame: shape=%s, %d columns", df.shape, df.shape[1])
    return df


def stage1_preprocess(
    df,
    sampling_fraction=SAMPLING_FRACTION,
    target_col=TARGET_COL,
    threshold=RARE_CLASS_THRESHOLD,
    random_state=RANDOM_STATE,
):
    """Stage 1 — Data Preprocessing (mirrors train_fair.py Steps 1-2)."""
    logging.info("=== Stage 1: Data Preprocessing ===")

    if target_col not in df.columns:
        msg = (f"Target column '{target_col}' not found. "
               f"Columns found: {list(df.columns)}")
        logging.error(msg)
        raise KeyError(msg)

    # 1. Subsample to the ~10k working set (identical to train_fair.py Step 1).
    logging.info("Shape BEFORE sampling: %s", df.shape)
    df = df.sample(frac=sampling_fraction, random_state=random_state).reset_index(drop=True)
    logging.info("Shape AFTER sampling %.4f%%: %s", sampling_fraction * 100, df.shape)

    # 2. Drop rows with a blank target label, then normalize label dtype.
    initial = df.shape[0]
    df = df.dropna(subset=[target_col])
    logging.info("Dropped %d rows with NaN in '%s'. Remaining: %d",
                 initial - df.shape[0], target_col, df.shape[0])
    df[target_col] = df[target_col].astype(str)

    # 3. Drop rare classes (keep labels with count >= threshold).
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= threshold].index
    logging.info("Class counts BEFORE filtering: %s", counts.to_dict())
    logging.info("Keeping classes with >= %d samples: %s",
                 threshold, list(valid_classes))
    df = df[df[target_col].isin(valid_classes)].reset_index(drop=True)

    if df.shape[0] < 2:
        msg = "Not enough data after filtering by class frequency."
        logging.error(msg)
        raise ValueError(msg)
    logging.info("Shape AFTER filtering rare classes: %s", df.shape)
    return df


def stage2_feature_select(
    df,
    selected_features=SELECTED_FEATURES,
    target_col=TARGET_COL,
    protected_col=PROTECTED_COL,
):
    """Stage 2 — Feature Selection (mirrors train_fair.py Step 4 + the
    feature-cleaning half of Step 6; no one-hot encoding)."""
    logging.info("=== Stage 2: Feature Selection ===")

    # 1. De-duplicate and intersect with columns actually present.
    unique_features = list(set(selected_features))
    final_features = [c for c in unique_features if c in df.columns]
    logging.info("Candidate features (unique): %s", unique_features)
    logging.info("Final features present in data (%d): %s",
                 len(final_features), final_features)
    if not final_features:
        msg = "No valid features found in the dataset. Adjust SELECTED_FEATURES."
        logging.error(msg)
        raise ValueError(msg)

    # 2. Keep features + target + protected column (retain 학교명 for fairness).
    keep = list(final_features)
    if target_col not in keep:
        keep.append(target_col)
    if protected_col in df.columns and protected_col not in keep:
        keep.append(protected_col)
    else:
        if protected_col not in df.columns:
            logging.warning("Protected column '%s' not in data; skipping it.",
                            protected_col)

    out = df[keep].copy()

    # 3. Clean the feature columns: fillna("missing") + cast to str.
    for c in final_features:
        out[c] = out[c].fillna("missing").astype(str)

    logging.info("Stage 2 output shape: %s (columns: %s)",
                 out.shape, list(out.columns))
    return out


def main():
    df_raw = load_raw_excel()

    df_stage1 = stage1_preprocess(df_raw)
    df_stage1.to_csv(STAGE1_CSV, index=False, encoding="utf-8-sig")
    logging.info("Wrote Stage 1 output -> '%s' (shape=%s)", STAGE1_CSV, df_stage1.shape)

    df_stage2 = stage2_feature_select(df_stage1)
    df_stage2.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logging.info("Wrote Stage 2 output -> '%s' (shape=%s)", OUTPUT_CSV, df_stage2.shape)

    # Concise summary for the console.
    print("\n=== Preprocessing complete ===")
    print(f"Stage 1 rows (after sample + dropna + rare-class filter): {df_stage1.shape[0]}")
    print(f"Stage 2 shape (feature-selected): {df_stage2.shape}")
    print(f"Stage 2 columns: {list(df_stage2.columns)}")
    print("\n최종학력 distribution (Stage 2):")
    print(df_stage2[TARGET_COL].value_counts().to_string())


if __name__ == "__main__":
    main()
