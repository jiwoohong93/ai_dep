"""Locate(causal tracing)용 최소쌍 생성기.

rule마다 10,000쌍: 동일 문장 2개가 오직 해당 rule 값만 현 threshold(θ) 경계 반대편.
값은 학습 gaussian 분포 N(θ, σ)를 재현해 in-distribution 영역에서 샘플링한다.
나머지 rule 지표는 통과(안전존) 값을 랜덤 샘플링, 프롬프트는 format_prompt 단일 소스.

출력: data/rules/edited_rule_25pct/locate/rule{1..9}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.check_rule_labels import compute_rule_labels
from src.models.test import format_prompt


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

# (case, column, theta, sigma, pass_side, delta, is_amount)
# sigma: 학습 gaussian 생성기와 동일 — 퍼센트 max(0.15*theta,1)/2, 금액 500,000
RULE_SPECS = [
    (1, "부채비율(%)", 200.0, 15.0, "below", 4.0, False),
    (2, "유동비율(%)", 100.0, 7.5, "above", 2.0, False),
    (3, "자기자본비율(%)", 20.0, 1.5, "above", 0.4, False),
    (4, "자본총계", 0.0, 500_000.0, "above", 10_000.0, True),
    (5, "보통주자본비율(%)", 4.5, 0.5, "above", 0.09, False),
    (6, "기본자본비율(%)", 6.0, 0.5, "above", 0.12, False),
    (7, "총자본비율(%)", 8.0, 0.6, "above", 0.16, False),
    (8, "단순기본자본비율(%)", 3.0, 0.5, "above", 0.06, False),
    (9, "공제후자본양수", 0.0, 500_000.0, "above", 10_000.0, True),
]

# 컨텍스트 (학습 gaussian 생성기 재현):
#   - 타겟 rule 제외 나머지 지표는 학습 때의 고정값(퍼센트=θ, 금액=1)
#   - 단, 행마다 파트너 지표 1개를 랜덤 선택해 N(θ_p, σ_p)로 흔듦 (학습의 지표쌍 순환 재현)
#   - 파트너가 rule 지표면 라벨 오염 방지를 위해 통과측으로 절단 (distractor는 자유)
AMOUNT_COLUMNS = {"자본총계", "공제후자본양수", "영업흑자", "이익잉여금양수"}

CONTEXT_FILL = {
    "보통주자본비율(%)": 4.5,
    "기본자본비율(%)": 6.0,
    "총자본비율(%)": 8.0,
    "단순기본자본비율(%)": 3.0,
    "공제후자본양수": 1,
    "총자산순이익률ROA(%)": 0.3,
    "영업흑자": 1,
    "이익잉여금양수": 1,
    "예대율(%)": 100.0,
    "부채비율(%)": 200.0,
    "유동비율(%)": 100.0,
    "자기자본비율(%)": 20.0,
    "자본총계": 1,
}

# 파트너 흔들기 중심/폭 — generate_train_gaussian.py의 THRESHOLDS·std 산식과 동일
PARTNER_CENTER = {
    **{col: theta for _, col, theta, *_ in RULE_SPECS},
    "총자산순이익률ROA(%)": 0.3,
    "영업흑자": 0.0,
    "이익잉여금양수": 0.0,
    "예대율(%)": 100.0,
}
PARTNER_SIGMA = {
    col: (500_000.0 if col in AMOUNT_COLUMNS else max(abs(center) * 0.15, 1) / 2)
    for col, center in PARTNER_CENTER.items()
}
# 파트너가 rule 지표일 때 지켜야 할 통과 조건 (반올림 후 값 기준)
RULE_PASS_CHECK = {
    "부채비율(%)": lambda v: v <= 200,
    "유동비율(%)": lambda v: v >= 100,
    "자기자본비율(%)": lambda v: v >= 20,
    "자본총계": lambda v: v > 0,
    "보통주자본비율(%)": lambda v: v >= 4.5,
    "기본자본비율(%)": lambda v: v >= 6.0,
    "총자본비율(%)": lambda v: v >= 8.0,
    "단순기본자본비율(%)": lambda v: v >= 3.0,
    "공제후자본양수": lambda v: v > 0,
}


def sample_partner_values(rng, column, size):
    """파트너 지표를 학습 분포 N(θ_p, σ_p)로 흔든다. rule 지표는 통과측만."""
    center, sigma = PARTNER_CENTER[column], PARTNER_SIGMA[column]
    check = RULE_PASS_CHECK.get(column)
    values = np.empty(size)
    filled = 0
    while filled < size:
        draw = rng.normal(center, sigma, size * 2)
        draw = draw.round(0) if column in AMOUNT_COLUMNS else draw.round(2)
        if check is not None:
            draw = draw[check(draw)]
        if column not in AMOUNT_COLUMNS:
            draw = draw[draw > 0]
        take = min(len(draw), size - filled)
        values[filled : filled + take] = draw[:take]
        filled += take
    return values


def sample_context(rng, exclude_column, size):
    """θ-고정 컨텍스트 + 행마다 파트너 1개 흔들기 (학습 gaussian 방식 재현)."""
    context = {
        column: np.full(size, float(fill))
        for column, fill in CONTEXT_FILL.items()
        if column != exclude_column
    }
    candidates = [c for c in context]
    partner_pick = rng.integers(0, len(candidates), size)
    partner_names = np.empty(size, dtype=object)
    for j, column in enumerate(candidates):
        mask = partner_pick == j
        if mask.any():
            context[column][mask] = sample_partner_values(rng, column, int(mask.sum()))
            partner_names[mask] = column
    for column in context:
        if column in AMOUNT_COLUMNS:
            context[column] = context[column].round().astype("int64")
        else:
            context[column] = context[column].round(2)
    context["_partner"] = partner_names
    return context


def sample_side(rng, theta, sigma, side, delta, is_amount, size):
    """N(theta, sigma)를 해당 측(신뢰 마진 delta 밖)으로 절단 샘플링."""
    values = np.empty(size)
    filled = 0
    while filled < size:
        draw = rng.normal(theta, sigma, size * 2)
        # 퍼센트 지표만 양수 제약 (금액 rule 4·9의 탈락측은 음수가 정상)
        lower_bound = -np.inf if is_amount else 0
        if side == "below":
            draw = draw[(draw < theta - delta) & (draw > lower_bound)]
        else:
            draw = draw[(draw > theta + delta) & (draw > lower_bound)]
        take = min(len(draw), size - filled)
        values[filled : filled + take] = draw[:take]
        filled += take
    if is_amount:
        return values.round().astype("int64")
    return values.round(2)


def build_pair_frames(rng, spec, size):
    case, column, theta, sigma, pass_side, delta, is_amount = spec
    fail_side = "above" if pass_side == "below" else "below"

    context = sample_context(rng, column, size)
    partner = pd.Series(context.pop("_partner"), name="partner")
    base = pd.DataFrame(context)
    base[column] = sample_side(rng, theta, sigma, pass_side, delta, is_amount, size)
    source = base.copy()
    source[column] = sample_side(rng, theta, sigma, fail_side, delta, is_amount, size)
    return base[FEATURE_COLUMNS], source[FEATURE_COLUMNS], partner


def count_tokens(tokenizer, texts, batch_size=512):
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size])
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.array(lengths)


def main():
    parser = argparse.ArgumentParser(description="Generate minimal pairs for rule locating.")
    parser.add_argument("--output_dir", default="data/rules/edited_rule_25pct/locate")
    parser.add_argument("--num_pairs", type=int, default=10_000)
    parser.add_argument("--resample_rounds", type=int, default=3)
    parser.add_argument("--tokenizer_name", default="yongukpark/xlm-roberta-large-rule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in RULE_SPECS:
        case, column, theta, sigma, pass_side, delta, is_amount = spec
        rng = np.random.default_rng(args.seed + case)

        base_df, source_df, partner = build_pair_frames(rng, spec, args.num_pairs)

        # 토큰 길이 정렬: 불일치 쌍은 rule 값만 재샘플링 (컨텍스트 유지)
        base_texts = base_df.apply(format_prompt, axis=1).tolist()
        source_texts = source_df.apply(format_prompt, axis=1).tolist()
        base_lengths = count_tokens(tokenizer, base_texts)
        source_lengths = count_tokens(tokenizer, source_texts)
        aligned = base_lengths == source_lengths
        for _ in range(args.resample_rounds):
            if aligned.all():
                break
            retry = np.flatnonzero(~aligned)
            base_df.loc[base_df.index[retry], column] = sample_side(
                rng, theta, sigma, pass_side, delta, is_amount, len(retry)
            )
            fail_side = "above" if pass_side == "below" else "below"
            source_df.loc[source_df.index[retry], column] = sample_side(
                rng, theta, sigma, fail_side, delta, is_amount, len(retry)
            )
            retry_base = base_df.iloc[retry].apply(format_prompt, axis=1).tolist()
            retry_source = source_df.iloc[retry].apply(format_prompt, axis=1).tolist()
            aligned[retry] = count_tokens(tokenizer, retry_base) == count_tokens(
                tokenizer, retry_source
            )

        # 라벨 무결성: base=1, source=0 (현 rule 기준)
        base_labels = compute_rule_labels(base_df)
        source_labels = compute_rule_labels(source_df)
        if not (base_labels == 1).all():
            raise AssertionError(f"rule {case}: base 라벨에 0이 존재 (컨텍스트 샘플링 버그)")
        if not (source_labels == 0).all():
            raise AssertionError(f"rule {case}: source 라벨에 1이 존재")

        out = base_df.copy()
        out.insert(0, "pair_id", np.arange(len(out)))
        out["value_fail"] = source_df[column].values
        out["base_label"] = base_labels.values
        out["source_label"] = source_labels.values
        out["token_aligned"] = aligned
        out["partner"] = partner.values

        output_path = output_dir / f"rule{case}.csv"
        out.to_csv(output_path, index=False)
        print(
            f"rule {case} ({column}): {len(out):,}쌍 | "
            f"token_aligned {aligned.mean()*100:.1f}% | saved: {output_path}"
        )


if __name__ == "__main__":
    main()
