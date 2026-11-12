import argparse
from pathlib import Path

import pandas as pd

try:
    from check_rule_labels import compute_rule_labels
except ModuleNotFoundError:
    from src.check_rule_labels import compute_rule_labels


BANK_COLUMNS = [
    "bank",
    "보통주자본비율(%)",
    "기본자본비율(%)",
    "총자본비율(%)",
    "단순기본자본비율(%)",
    "공제후자본양수",
    "총자산순이익률ROA(%)",
    "영업흑자",
    "이익잉여금양수",
    "예대율(%)",
]

FIRM_COLUMNS = [
    "company",
    "부채비율(%)",
    "유동비율(%)",
    "자기자본비율(%)",
    "자본총계",
]

OUTPUT_COLUMNS = (
    ["bank", "company"]
    + [col for col in BANK_COLUMNS if col != "bank"]
    + [col for col in FIRM_COLUMNS if col != "company"]
    + ["label"]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create every bank-company pair and label it with check_rule_labels.py rules."
    )
    parser.add_argument(
        "--bank_path",
        default="data/original/text/bank_labels_v5.csv",
        help="Input bank CSV path.",
    )
    parser.add_argument(
        "--firm_path",
        default="data/original/text/firm_labels_v3.csv",
        help="Input firm CSV path.",
    )
    parser.add_argument(
        "--output_path",
        default="data/test/text/test.csv",
        help="Output test CSV path.",
    )
    return parser.parse_args()


def validate_columns(df, required_columns, path):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )


def load_selected_columns(path, columns):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, columns, csv_path)
    return df[columns].copy()


def build_pairs(bank_df, firm_df):
    bank_df = bank_df.copy()
    firm_df = firm_df.copy()
    bank_df["_join_key"] = 1
    firm_df["_join_key"] = 1

    pairs = bank_df.merge(firm_df, on="_join_key", how="inner").drop(columns="_join_key")
    pairs["label"] = compute_rule_labels(pairs)
    return pairs[OUTPUT_COLUMNS]


def main():
    args = parse_args()
    bank_df = load_selected_columns(args.bank_path, BANK_COLUMNS)
    firm_df = load_selected_columns(args.firm_path, FIRM_COLUMNS)

    pairs = build_pairs(bank_df, firm_df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_path, index=False)

    print(f"bank rows: {len(bank_df):,}")
    print(f"firm rows: {len(firm_df):,}")
    print(f"pair rows: {len(pairs):,}")
    print(f"label counts: {pairs['label'].value_counts().sort_index().to_dict()}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
