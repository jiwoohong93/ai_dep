"""Bias centroid 실험용 정렬(aligned) counterfactual 쌍 생성 -- train.csv만 사용, test.csv 무사용.

두 신호를 분리해서 만든다 (README "확정 목표": cg rule은 안 건드리고 비혼 표지만 완화):
- marital 쌍: 자격있는(label==1) 실제 행을 골라, relationship/marital-status만
  강제로 "기혼"/"비혼"으로 바꾼 두 버전 -- 같은 사람, 같은 나머지 13개 변수.
  (probe_neutral_relationship.py의 중립화 컨벤션과 동일: 기혼=Husband/Wife(성별 정합)+
   Married-civ-spouse, 비혼=Not-in-family+Never-married)
- cg 쌍: 실제 문맥 행을 골라 capital-gain만 7,000 경계 위/아래 값으로 바꾼 두 버전.
  (generate_cg_boundary_test.py와 동일한 theta=7000, sigma=525 가우시안 관행)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


import bias_centroid_common as bc  # noqa: E402


def sample_marital_pairs(train_df: pd.DataFrame, n: int, seed: int = 0):
    qualified = train_df[train_df["label"] == 1]
    base = qualified.sample(n=min(n, len(qualified)), random_state=seed).reset_index(drop=True)

    married = base.copy()
    married["relationship"] = np.where(married["sex"] == "Male", "Husband", "Wife")
    married["marital-status"] = "Married-civ-spouse"

    unmarried = base.copy()
    unmarried["relationship"] = "Not-in-family"
    unmarried["marital-status"] = "Never-married"

    return married.reset_index(drop=True), unmarried.reset_index(drop=True), base


def _sample_side(rng, n, side):
    out = []
    while len(out) < n:
        draw = rng.normal(bc.CG_THETA, bc.CG_SIGMA, size=n * 3)
        draw = draw[(draw < bc.CG_THETA) if side == "below" else (draw >= bc.CG_THETA)]
        draw = draw[draw >= 0]
        out.extend(np.round(draw).astype(int).tolist())
    return np.array(out[:n])


def sample_cg_pairs(train_df: pd.DataFrame, n: int, seed: int = 1):
    """실제 문맥(13개 변수, capital-gain 제외) 행을 뽑아 cg만 경계 위/아래로 교체."""
    base = train_df.sample(n=min(n, len(train_df)), random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    below = base.copy()
    below["capital-gain"] = _sample_side(rng, len(base), "below")

    above = base.copy()
    above["capital-gain"] = _sample_side(rng, len(base), "above")

    return above.reset_index(drop=True), below.reset_index(drop=True), base


NEW_CG_THETA = 7700.0  # rule(정책) 편집: 7,000 -> 7,700 (1,399명 중 544명/38.9% 발동해제)


def sample_cg_band_pairs(train_df: pd.DataFrame, n: int, seed: int = 0):
    """band(새 임계값 아래로 떨어지는 구간, [7000,7700)) 문맥과, 그 짝(below<7000 / above>=7700) 도너.

    band는 구rule(7000)에서는 발동(1)했지만 신rule(7700)에서는 더 이상 발동하면 안 되는 구간 --
    Regulation의 85~200% band와 동일한 위상. base(문맥, capital-gain 제외 13개)는 train.csv에서
    뽑은 실제 행이라 다양성이 보장된다(v1의 좁은 합성 문맥 문제 재발 방지).
    """
    rng = np.random.default_rng(seed)
    base = train_df.sample(n=min(n, len(train_df)), random_state=seed).reset_index(drop=True)

    band = base.copy()
    band["capital-gain"] = rng.integers(7000, 7700, size=len(base))

    below = base.copy()
    below["capital-gain"] = rng.integers(0, 5000, size=len(base))

    above = base.copy()
    above["capital-gain"] = rng.integers(8000, 15000, size=len(base))

    return band.reset_index(drop=True), below.reset_index(drop=True), above.reset_index(drop=True)


if __name__ == "__main__":
    # 조립 문장이 원본 format_prompt와 문자 단위로 일치하는지 1회 검증
    train = bc.load_train_csv()
    for _, row in train.sample(20, random_state=0).iterrows():
        bc._assert_matches_reference(row)
    print("format_prompt 세그먼트 조립 검증 통과 (20/20 행)")

    married, unmarried, _ = sample_marital_pairs(train, n=5, seed=0)
    print("\n[marital 쌍 예시]")
    print("married:", bc.build_prompt_with_spans(married.iloc[0])[0])
    print("unmarried:", bc.build_prompt_with_spans(unmarried.iloc[0])[0])

    above, below, _ = sample_cg_pairs(train, n=5, seed=1)
    print("\n[cg 쌍 예시]")
    print("above:", bc.build_prompt_with_spans(above.iloc[0])[0])
    print("below:", bc.build_prompt_with_spans(below.iloc[0])[0])
