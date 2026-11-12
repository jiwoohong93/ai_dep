import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from check_rule_labels import RULES, compute_rule_labels


FEATURE_COLUMNS = [
    "보통주자본비율(%)",
    "기본자본비율(%)",
    "총자본비율(%)",
    "단순기본자본비율(%)",
    "공제후자본양수",
    "총자산순이익률ROA(%)",
    "영업흑자",
    "이익잉여금양수",
    "예대율(%)",
    "부채비율(%)",
    "유동비율(%)",
    "자기자본비율(%)",
    "자본총계",
]
THRESHOLDS = {
    **{column: threshold for column, _, threshold in RULES},
    "총자산순이익률ROA(%)": 0.3,
    "영업흑자": 0,
    "이익잉여금양수": 0,
    "예대율(%)": 100,
}
AMOUNT_COLUMNS = ["공제후자본양수", "영업흑자", "이익잉여금양수", "자본총계"]
PERCENT_COLUMNS = [column for column in FEATURE_COLUMNS if column not in AMOUNT_COLUMNS]
OUTPUT_COLUMNS = ["bank", "company", *FEATURE_COLUMNS, "label"]


def generate_dataset(num_rows, seed):
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame(
        {
            column: np.full(
                num_rows,
                1 if column in AMOUNT_COLUMNS else threshold,
                dtype=float,
            )
            for column, threshold in THRESHOLDS.items()
        }
    )

    feature_pairs = list(combinations(FEATURE_COLUMNS, 2))
    for pair_index, pair in enumerate(feature_pairs):
        indices = np.arange(pair_index, num_rows, len(feature_pairs))
        for column in pair:
            threshold = THRESHOLDS[column]
            if column in AMOUNT_COLUMNS:
                std = 500_000
            else:
                std = max(abs(threshold) * 0.15, 1) / 2
            rows.loc[indices, column] = rng.normal(threshold, std, len(indices))

    rows[PERCENT_COLUMNS] = rows[PERCENT_COLUMNS].round(2)
    rows[AMOUNT_COLUMNS] = rows[AMOUNT_COLUMNS].round().astype("int64")
    rows.insert(0, "company", [f"firm_{index:07d}" for index in range(num_rows)])
    rows.insert(0, "bank", [f"bank_{index:07d}" for index in range(num_rows)])
    rows["label"] = compute_rule_labels(rows)
    return rows[OUTPUT_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description="Create training data around the rule thresholds."
    )
    parser.add_argument(
        "--output_path",
        default="data/train/text/train_gaussian.csv",
    )
    parser.add_argument("--num_rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_rows < 1:
        parser.error("--num_rows must be positive")

    df = generate_dataset(args.num_rows, args.seed)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"output rows: {len(df):,}")
    print(f"label counts: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
