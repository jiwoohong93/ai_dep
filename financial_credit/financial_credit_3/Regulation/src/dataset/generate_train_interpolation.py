"""입력 CSV의 행을 그대로 쓰지 않고 두 행의 볼록결합(α·A+(1-α)·B)으로 새 합성행을 만들며,
라벨은 규정(check_rule_labels)으로 재계산한다. 시험 인스턴스와의 중복은
docs/train_test_overlap.md 참고 (기본 입력 1,000,000행 기준 13 feature 전체 일치 0건).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_rule_labels import compute_rule_labels


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
INTERPOLATED_COLUMNS = [column for column in FEATURE_COLUMNS if column != "예대율(%)"]
PERCENT_COLUMNS = [column for column in INTERPOLATED_COLUMNS if "(%)" in column]
AMOUNT_COLUMNS = [column for column in INTERPOLATED_COLUMNS if "(%)" not in column]
OUTPUT_COLUMNS = ["bank", "company", *FEATURE_COLUMNS, "label"]
CHUNK_SIZE = 200_000


def load_source(path):
    source = pd.read_csv(path, usecols=INTERPOLATED_COLUMNS, low_memory=False)
    source = source.apply(pd.to_numeric, errors="coerce").dropna()
    if source.empty:
        raise ValueError("Input has no complete numeric rows to interpolate.")
    return source.to_numpy(dtype=float)


def generate_chunk(source, start, size, rng, loan_deposit_values):
    first = rng.integers(len(source), size=size)
    second = rng.integers(len(source), size=size)
    alpha = rng.random(size)
    mixed = alpha[:, None] * source[first] + (1 - alpha[:, None]) * source[second]

    rows = pd.DataFrame(mixed, columns=INTERPOLATED_COLUMNS)
    rows[PERCENT_COLUMNS] = rows[PERCENT_COLUMNS].round(2)
    rows[AMOUNT_COLUMNS] = rows[AMOUNT_COLUMNS].round(1)

    end = start + size
    rows.insert(8, "예대율(%)", loan_deposit_values[start:end])
    rows.insert(0, "company", [f"firm_{index:07d}" for index in range(start, end)])
    rows.insert(0, "bank", [f"bank_{index:07d}" for index in range(start, end)])
    rows["label"] = compute_rule_labels(rows)
    return rows[OUTPUT_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description="Create training data by interpolating test rows."
    )
    parser.add_argument("--input_path", default="data/test/text/test.csv")
    parser.add_argument(
        "--output_path",
        default="data/train/text/train_interpolated.csv",
    )
    parser.add_argument("--num_rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_rows < 1:
        parser.error("--num_rows must be positive")

    source = load_source(args.input_path)
    rng = np.random.default_rng(args.seed)

    loan_deposit_values = np.clip(
        rng.normal(100, 8, args.num_rows), 70, 130
    ).round(2).astype(object)
    unavailable_count = round(args.num_rows * 0.05)
    if unavailable_count:
        unavailable = rng.choice(args.num_rows, unavailable_count, replace=False)
        loan_deposit_values[unavailable] = "산정불가"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = {0: 0, 1: 0}

    for start in range(0, args.num_rows, CHUNK_SIZE):
        size = min(CHUNK_SIZE, args.num_rows - start)
        chunk = generate_chunk(source, start, size, rng, loan_deposit_values)
        counts = chunk["label"].value_counts()
        label_counts[0] += int(counts.get(0, 0))
        label_counts[1] += int(counts.get(1, 0))
        chunk.to_csv(
            output_path,
            mode="w" if start == 0 else "a",
            header=start == 0,
            index=False,
        )

    print(f"source rows: {len(source):,}")
    print(f"output rows: {args.num_rows:,}")
    print(f"label counts: {label_counts}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
