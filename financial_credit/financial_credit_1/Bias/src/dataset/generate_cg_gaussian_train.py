"""cg >= 7000 rule 주입용 gaussian 학습 데이터 생성 (fixed 문맥).

Regulation의 rule 경계 gaussian 학습 데이터의 대응물:
- capital-gain ~ N(7000, 525) 양측 균형 샘플, label = (cg >= 7000)
- 문맥(나머지 13개 feature)은 train label-0 대표 프로필로 고정
  (경계 진단셋과 동일 프로필 — cg만이 라벨을 결정하는 구성)
- 단, sex는 고정하지 않고 **라벨(경계 아래/위) 각각에서 남:여 = 50:50 정확 균형**
  — rule이 성별과 독립임을 데이터가 직접 보이게 함 (rule 주입이 성별 편향을
  새로 만들거나 남성 전용 rule로 학습되는 것을 방지)
- --balance sex_relationship: 라벨별로 다음 4셀을 25%씩 정확 균형 —
  (남, Husband, 기혼) / (여, Wife, 기혼) / (남, Not-in-family, 비혼) /
  (여, Not-in-family, 비혼). sex 50:50에 더해 **Husband/Wife(기혼 프록시)도
  라벨과 독립**임을 가르침. marital-status를 relationship과 정합시켜
  모순 문장("비혼인데 Husband")을 방지.
- 기본 16,280행(실데이터 32,561의 절반) → 배치 2:1(실:생성) 혼합용

주의: 문맥이 단일 프로필이므로 rule 일반화 평가는 random 문맥
경계셋(cg_boundary_random.csv)이 held-out OOD 프로브 역할을 한다.

출력: Bias/data/train_cg_mix/text/cg_gaussian.csv (기존 data 폴더는 불변)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import FEATURE_COLUMNS
from src.dataset.generate_cg_boundary_test import THETA, label0_profile, sample_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", choices=["sex", "sex_relationship"], default="sex")
    parser.add_argument("--n_per_side", type=int, default=8140)
    parser.add_argument("--seed", type=int, default=7,
                        help="경계 진단셋(seed 42)과 다른 시드로 cg 값 중복 최소화")
    parser.add_argument("--train_path", default=str(BIAS_ROOT / "data/train/text/train.csv"))
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()
    if args.output_path is None:
        name = "cg_gaussian.csv" if args.balance == "sex" else "cg_gaussian_relbal.csv"
        args.output_path = str(BIAS_ROOT / "data/train_cg_mix/text" / name)
    rng = np.random.default_rng(args.seed)

    train = pd.read_csv(args.train_path)
    profile = label0_profile(train)
    print("fixed 문맥 프로필:", profile)

    n_total = args.n_per_side * 2
    df = pd.DataFrame({col: [v] * n_total for col, v in profile.items()})
    df["capital-gain"] = np.concatenate([
        sample_side(rng, args.n_per_side, "below"),
        sample_side(rng, args.n_per_side, "above"),
    ])
    if args.balance == "sex":
        # 라벨(side)별 남:여 = 50:50 정확 균형
        half = args.n_per_side // 2
        side_sex = ["Male"] * half + ["Female"] * (args.n_per_side - half)
        df["sex"] = side_sex + side_sex
    else:
        # 라벨(side)별 4셀 25% 균형: (남,Husband,기혼)/(여,Wife,기혼)/
        # (남,Not-in-family,비혼)/(여,Not-in-family,비혼)
        cells = [
            ("Male", "Husband", "Married-civ-spouse"),
            ("Female", "Wife", "Married-civ-spouse"),
            ("Male", "Not-in-family", "Never-married"),
            ("Female", "Not-in-family", "Never-married"),
        ]
        q, r = divmod(args.n_per_side, len(cells))
        side = []
        for i, cell in enumerate(cells):
            side.extend([cell] * (q + (1 if i < r else 0)))
        both = side + side
        df["sex"] = [c[0] for c in both]
        df["relationship"] = [c[1] for c in both]
        df["marital-status"] = [c[2] for c in both]
    df["label"] = (df["capital-gain"] >= THETA).astype(int)
    df = df[FEATURE_COLUMNS + ["label"]]
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    below, above = df[df.label == 0]["capital-gain"], df[df.label == 1]["capital-gain"]
    print(f"saved: {out} ({len(df):,}행)")
    print(f"below: {len(below):,}행 cg[{below.min()},{below.max()}] | "
          f"above: {len(above):,}행 cg[{above.min()},{above.max()}]")


if __name__ == "__main__":
    main()
