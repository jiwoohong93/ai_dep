"""EDRM centroid v2 — 순수 합성 데이터 파이프라인 공통 유틸.

v1(centroid_common.py 등)의 근본 문제: locate·delta_v·key·covariance를 실제 test.csv에서
계산했는데, Exp5의 최종 평가도 같은 test.csv(전체)로 했다 -- 편집 방향을 만든 데이터와
채점하는 데이터가 겹쳐 과적합/데이터 누수였다 (docs/edit_baseline_plan.md의 명시적 금지 위반).

v2는 원칙을 강제한다:
  - locate, delta_v, key, covariance, gate 학습/보정: data/rules/edited_rule_25pct/locate/
    의 합성 최소쌍(rule당 10,000개, 실제 은행/기업 데이터 전혀 없음)과 그로부터 파생한
    추가 합성 그룹만 사용
  - test.csv(실데이터): Exp5 최종 평가에만 사용 (여기서만 import됨)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


import centroid_common as cc  # noqa: E402
from src.editing.generate_locate_pairs import FEATURE_COLUMNS, RULE_SPECS, AMOUNT_COLUMNS  # noqa: E402

RESULT_ROOT_V2 = cc.CENTROID_ROOT / "result" / "centroid_v2"
CACHE_DIR_V2 = RESULT_ROOT_V2 / "cache"
LOCATE_PAIRS_DIR = cc.REGULATION_ROOT / "data" / "rules" / "edited_rule_25pct" / "locate"


def ensure_dirs_v2():
    for d in (RESULT_ROOT_V2, CACHE_DIR_V2):
        d.mkdir(parents=True, exist_ok=True)


def load_locate_pairs(rule_num: int, aligned_only=True):
    """합성 최소쌍 로드. base_df(통과)/source_df(탈락) 둘 다 FEATURE_COLUMNS 스키마로 반환
    (기존 cc.extract_layer_position_hidden 등 v1 유틸을 그대로 재사용할 수 있게)."""
    _, rule_column, theta, *_ = cc.rule_spec(rule_num)
    path = LOCATE_PAIRS_DIR / f"rule{rule_num}.csv"
    df = pd.read_csv(path)
    if aligned_only:
        df = df[df["token_aligned"]].reset_index(drop=True)
    assert (df["base_label"] == 1).all() and (df["source_label"] == 0).all(), \
        "합성쌍의 base/source 라벨이 기대(1/0)와 다름 -- 데이터 손상 의심"

    base_df = df[FEATURE_COLUMNS].copy()
    source_df = base_df.copy()
    if rule_column in AMOUNT_COLUMNS:
        source_df[rule_column] = df["value_fail"].astype(int).values
    else:
        source_df[rule_column] = df["value_fail"].values
    return base_df, source_df, df


def synth_band_safe_fail(rule_num: int, new_threshold: float, n_templates=100, seed=0,
                          grid=None):
    """Exp3(v2)용: 부채비율 그리드 sweep이지만 컨텍스트가 실템플릿이 아니라 locate pair의
    합성 컨텍스트(파트너-사이클) 방식을 그대로 재사용 -- 100% 합성."""
    from src.editing.generate_locate_pairs import (
        sample_context, CONTEXT_FILL,
    )
    _, rule_column, theta, sigma, pass_side, delta, is_amount = cc.rule_spec(rule_num)
    if grid is None:
        grid = np.arange(70, 221, 5)
    rng = np.random.default_rng(seed)

    rows = []
    for t_id in range(n_templates):
        context = sample_context(rng, rule_column, 1)
        partner = context.pop("_partner")[0]
        base_row = {col: float(context[col][0]) for col in context}
        for v in grid:
            row = dict(base_row)
            row[rule_column] = round(float(v), 1)
            row["template_id"] = t_id
            row["partner"] = partner
            rows.append(row)
    synth = pd.DataFrame(rows).reset_index(drop=True)
    for col in AMOUNT_COLUMNS:
        if col in synth.columns and col != rule_column:
            synth[col] = synth[col].round().astype("int64")
    synth["value_grid"] = synth[rule_column].astype(float)
    synth["band"] = np.where(synth["value_grid"] <= new_threshold, "safe",
                      np.where(synth["value_grid"] <= theta, "band", "fail_old"))
    return synth, rule_column


def synth_other_rule_fail(rule_num: int, n=500, seed=1):
    """target rule(rule_num) 자체는 안전존(통과)이지만, 다른 rule 중 하나를 일부러
    탈락시켜 만든 합성 샘플 ("타rule 탈락" keep-set, Exp4.0 covariance/AUC용).
    generate_locate_pairs.py의 다른 rule들의 RULE_SPECS를 이용해 그 rule을 탈락측으로
    샘플링하고, target rule과 나머지는 안전존 고정값으로 채운다."""
    from src.editing.generate_locate_pairs import CONTEXT_FILL, sample_side
    _, target_column, target_theta, _, target_pass_side, target_delta, target_is_amount = cc.rule_spec(rule_num)
    other_specs = [s for s in RULE_SPECS if s[1] != target_column]
    rng = np.random.default_rng(seed)

    rows = []
    per_rule = max(1, n // len(other_specs))
    for spec in other_specs:
        case, column, theta, sigma, pass_side, delta, is_amount = spec
        fail_side = "above" if pass_side == "below" else "below"
        row_template = {col: float(fill) for col, fill in CONTEXT_FILL.items()}
        fail_vals = sample_side(rng, theta, sigma, fail_side, delta, is_amount, per_rule)
        for v in fail_vals:
            row = dict(row_template)
            row[column] = v
            row["_failed_rule"] = column
            rows.append(row)
    df = pd.DataFrame(rows).reset_index(drop=True)
    for col in AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("int64")
    return df[FEATURE_COLUMNS + ["_failed_rule"]]


def synth_generic(n=1000, seed=2):
    """ROME의 '위키텍스트' 대응물 -- 특정 목적 없이 여러 rule의 안전존/탈락존을 섞은
    넓은 합성 표본 (여러 rule의 파트너 사이클링 컨텍스트를 랜덤하게 조합)."""
    from src.editing.generate_locate_pairs import sample_context
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rule_num = rng.integers(1, 10)
        _, rule_column, theta, sigma, pass_side, delta, is_amount = cc.rule_spec(int(rule_num))
        context = sample_context(rng, rule_column, 1)
        context.pop("_partner")
        row = {col: float(context[col][0]) for col in context}
        # target rule 자체도 안전측 값으로 채움 (특정 목적 없는 "일반" 샘플)
        from src.editing.generate_locate_pairs import sample_side
        row[rule_column] = float(sample_side(rng, theta, sigma, pass_side, delta, is_amount, 1)[0])
        rows.append(row)
    df = pd.DataFrame(rows).reset_index(drop=True)
    for col in AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("int64")
    return df[FEATURE_COLUMNS]
