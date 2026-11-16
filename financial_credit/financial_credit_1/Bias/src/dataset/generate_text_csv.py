"""UCI Adult 원본 → 학습/평가 CSV 생성.

- train.csv: adult.data(32,561행) 전체 — KAIST baseline과 동일하게 val 분할 없음
  (에폭별 test 측정으로 범위/최종에폭을 보고하는 재측정 프로토콜과 대칭)
- test.csv: adult.test(16,281행) 전체
- 결측 '?'는 'unknown'으로 치환 (행 삭제 안 함 — baseline과 동일 행수 유지)
- 수치 컬럼은 정수로 정규화 (adult.test의 float 표기 정리)
- label = (income == '>50K') -> 1

출력: Bias/data/train/text/train.csv, Bias/data/test/text/test.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import FEATURE_COLUMNS

RAW = BIAS_ROOT / "data/raw/adult"
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ORIG_ORDER = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
              "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
              "hours-per-week", "native-country", "income"]


def load_raw(path, skiprows=0):
    df = pd.read_csv(path, names=ORIG_ORDER, skipinitialspace=True, skiprows=skiprows)
    df["income"] = df["income"].str.rstrip(".")
    assert set(df["income"].unique()) == {">50K", "<=50K"}, df["income"].unique()
    df["label"] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["income"]).replace("?", "unknown")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col]).astype(np.int64)
    assert df.notna().all().all()
    return df[FEATURE_COLUMNS + ["label"]]


def main():
    train = load_raw(RAW / "adult.data")
    test = load_raw(RAW / "adult.test", skiprows=1)
    assert len(train) == 32561 and len(test) == 16281

    out_train = BIAS_ROOT / "data/train/text"
    out_test = BIAS_ROOT / "data/test/text"
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train / "train.csv", index=False)
    test.to_csv(out_test / "test.csv", index=False)

    for name, df in [("train", train), ("test", test)]:
        print(f"{name}: {len(df):,}행, P(label=1)={df['label'].mean():.4f}")


if __name__ == "__main__":
    main()
