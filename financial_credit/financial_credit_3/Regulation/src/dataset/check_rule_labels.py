import argparse
import operator
from pathlib import Path

import pandas as pd


RULES = [
    # company
    ("부채비율(%)", "<=", 200),
    ("유동비율(%)", ">=", 100),
    ("자기자본비율(%)", ">=", 20),
    ("자본총계", ">", 0),
    # bank
    ("보통주자본비율(%)", ">=", 4.5),
    ("기본자본비율(%)", ">=", 6.0),
    ("총자본비율(%)", ">=", 8.0),
    ("단순기본자본비율(%)", ">=", 3.0),
    ("공제후자본양수", ">", 0),
]

OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    ">": operator.gt,
}


def compute_rule_labels(df):
    passed = pd.Series(True, index=df.index)

    for column, op, threshold in RULES:
        values = pd.to_numeric(df[column], errors="coerce")
        passed &= OPS[op](values, threshold).fillna(False)

    return passed.astype(int)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether dataset labels match the fixed financial rules."
    )
    parser.add_argument("data_path", help="CSV dataset path.")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = [column for column, _, _ in RULES] + [args.label_col]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["rule_label"] = compute_rule_labels(df)
    df["_given_label"] = pd.to_numeric(df[args.label_col], errors="coerce")
    df["label_matches_rule"] = df["_given_label"].eq(df["rule_label"])
    mismatched_rows = df.loc[~df["label_matches_rule"]]

    print(f"rows={len(df):,}")
    print(f"matched={len(df) - len(mismatched_rows):,}")
    print(f"mismatched={len(mismatched_rows):,}")
    print(f"rule_label_counts={df['rule_label'].value_counts().sort_index().to_dict()}")

    if not mismatched_rows.empty:
        print("\nMismatched rows preview:")
        preview_columns = required_columns[:-1] + ["_given_label", "rule_label"]
        print(mismatched_rows[preview_columns].head(20))

    if not mismatched_rows.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
