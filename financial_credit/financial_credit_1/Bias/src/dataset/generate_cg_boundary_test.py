"""capital-gain >= 7000 soft rule 이해도 진단용 경계 test셋 생성.

Regulation의 gaussian 경계 데이터와 동일 철학:
- capital-gain ~ N(7000, sigma),  sigma = 0.15 * 7000 / 2 = 525 (Regulation 퍼센트 rule 관행)
  → 절반은 경계 아래(비발동존), 절반은 위(발동존)가 되도록 양측 균형 샘플
- label: rule 기준 — cg >= 7000 -> 1, 미만 -> 0

문맥(나머지 13개 feature) 모드 두 가지:
- fixed (기본): train의 label-0 행에서 계산한 대표 프로필(범주=최빈값, 수치=중앙값)로
  전 행 고정. Regulation의 "다른 rule=통과값 고정"의 대응 — cg=0이면 확실히 0인
  프로필(실데이터 검증 P(y=1)=0.009)이라 **cg만이 판정을 뒤집을 수 있는 변수**가 됨.
  랜덤 문맥의 교란 없이 rule 자체의 학습 여부를 측정.
- random: 각 feature를 train 주변분포에서 독립 샘플 (문맥 랜덤화 — 문맥-독립성 측정).
  주의: 아래쪽 0 라벨은 random 문맥에선 보장이 아님(실데이터 [5k,7k)는 P=0.45).

출력: data/test/text/cg_boundary_{fixed|random}.csv (기본 2,000행, 아래/위 각 1,000)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import FEATURE_COLUMNS

THETA = 7000
SIGMA = 0.15 * THETA / 2  # = 525


def sample_side(rng, n, side):
    """N(7000, 525)에서 side('below'/'above')에 속하는 값 n개 (정수, >=0)."""
    out = []
    while len(out) < n:
        draw = rng.normal(THETA, SIGMA, size=n * 3)
        draw = draw[(draw < THETA) if side == "below" else (draw >= THETA)]
        out.extend(np.round(draw).astype(int).tolist())
    return np.array(out[:n])


NUMERIC_CTX = ["age", "fnlwgt", "education-num", "capital-loss", "hours-per-week"]


def label0_profile(train):
    """label-0 집단 대표 프로필: 범주=최빈값, 수치=중앙값."""
    lo = train[train.label == 0]
    profile = {}
    for col in FEATURE_COLUMNS:
        if col == "capital-gain":
            continue
        profile[col] = (int(lo[col].median()) if col in NUMERIC_CTX
                        else lo[col].mode()[0])
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--n_per_side", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_path", default=str(BIAS_ROOT / "data/train/text/train.csv"))
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = str(
            BIAS_ROOT / f"data/test/text/cg_boundary_{args.context_mode}.csv")
    rng = np.random.default_rng(args.seed)

    train = pd.read_csv(args.train_path)
    n_total = args.n_per_side * 2

    if args.context_mode == "fixed":
        profile = label0_profile(train)
        print("fixed 문맥 프로필:", profile)
        df = pd.DataFrame({col: [v] * n_total for col, v in profile.items()})
    else:
        # train 주변분포에서 독립 샘플 (행 단위 재사용 아님 → 상관 붕괴)
        df = pd.DataFrame({
            col: rng.choice(train[col].to_numpy(), size=n_total, replace=True)
            for col in FEATURE_COLUMNS if col != "capital-gain"
        })
    df["capital-gain"] = np.concatenate([
        sample_side(rng, args.n_per_side, "below"),
        sample_side(rng, args.n_per_side, "above"),
    ])
    df["label"] = (df["capital-gain"] >= THETA).astype(int)
    df = df[FEATURE_COLUMNS + ["label"]]
    df.to_csv(args.output_path, index=False)

    below = df[df.label == 0]["capital-gain"]
    above = df[df.label == 1]["capital-gain"]
    print(f"saved: {args.output_path} ({len(df):,}행)")
    print(f"below: {len(below)}행, cg [{below.min()}, {below.max()}]")
    print(f"above: {len(above)}행, cg [{above.min()}, {above.max()}]")


if __name__ == "__main__":
    main()
