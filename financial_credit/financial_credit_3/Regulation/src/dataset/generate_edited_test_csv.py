import argparse
from pathlib import Path

import pandas as pd

try:
    from check_rule_labels import OPS, RULES
except ModuleNotFoundError:
    from src.dataset.check_rule_labels import OPS, RULES


# data/rules/edited_rule_25pct.md 의 9개 케이스 (decisive 샘플의 ~25%가 1->0 플립되는 threshold)
EDITED_CASES = [
    (1, "부채비율(%)", 85),
    (2, "유동비율(%)", 140),
    (3, "자기자본비율(%)", 54),
    (4, "자본총계", 62_000_000_000),
    (5, "보통주자본비율(%)", 12.0),
    (6, "기본자본비율(%)", 12.5),
    (7, "총자본비율(%)", 12.5),
    (8, "단순기본자본비율(%)", 6.3),
    (9, "공제후자본양수", 3_000_000_000_000),
]


def compute_labels_with_edit(df, edited_column, new_threshold):
    passed = pd.Series(True, index=df.index)
    for column, op, threshold in RULES:
        if column == edited_column:
            threshold = new_threshold
        values = pd.to_numeric(df[column], errors="coerce")
        passed &= OPS[op](values, threshold).fillna(False)
    return passed.astype(int)


def main():
    parser = argparse.ArgumentParser(
        description="Copy test.csv with labels recomputed for each edited-rule case."
    )
    parser.add_argument("--input_path", default="data/test/text/test.csv")
    parser.add_argument("--output_dir", default="data/test/text/edited_rule_25pct")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, low_memory=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case, column, new_threshold in EDITED_CASES:
        edited = df.copy()
        edited["label"] = compute_labels_with_edit(df, column, new_threshold)
        flipped = (df["label"] != edited["label"]).sum()
        output_path = output_dir / f"test_case{case}.csv"
        edited.to_csv(output_path, index=False)
        print(
            f"case {case}: {column} -> {new_threshold:g} | "
            f"플립 {flipped:,} | label counts "
            f"{edited['label'].value_counts().sort_index().to_dict()} | {output_path}"
        )


if __name__ == "__main__":
    main()
