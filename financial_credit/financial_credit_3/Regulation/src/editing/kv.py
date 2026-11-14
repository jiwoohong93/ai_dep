"""편집(ROME/MEMIT/EMMET/AlphaEdit) 공통 재료 계산.

- 편집 요청/보존/일반 프롬프트 합성 (train/test 사용 금지 — locate 컨텍스트 로직 재사용)
- k  : 대상 층 MLP fc2 입력 (GELU(fc1(x))), value_last 토큰 위치
- v  : fc2 출력 (현재 값) / v* : 분류를 새 라벨로 뒤집는 목표 벡터 (경사하강 탐색)
- C  : 일반 프롬프트 전 위치 key의 2차 모멘트 E[kkᵀ]
- K₀ : 보존 키 (target rule 안전존 + 다른 rule 문맥) — AlphaEdit null-space용
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.generate_edited_test_csv import EDITED_CASES
from src.editing.generate_locate_pairs import (
    AMOUNT_COLUMNS,
    FEATURE_COLUMNS,
    RULE_SPECS,
    sample_context,
)
from src.editing.locate import build_prompt_with_spans, build_positions

CASE_SPECS = {}
for (case, column, theta, sigma, pass_side, delta, is_amount) in RULE_SPECS:
    new_threshold = next(t for c, col, t in EDITED_CASES if c == case)
    CASE_SPECS[case] = {
        "column": column, "theta": theta, "theta_new": new_threshold,
        "pass_side": pass_side, "is_amount": is_amount,
    }


def sample_zone_values(case, zone, n, rng):
    """flip: 현 rule 통과·새 rule 탈락 / safe: 둘 다 통과. 금액 rule은 log-uniform."""
    s = CASE_SPECS[case]
    theta, theta_new, is_amount = s["theta"], s["theta_new"], s["is_amount"]
    if s["pass_side"] == "below":  # ≤ rule (부채비율): flip = (θ′, θ], safe = (0, θ′)
        low, high = (theta_new * 1.01, theta) if zone == "flip" else (theta_new * 0.1, theta_new * 0.99)
    elif is_amount:  # > rule: flip = (0, θ′], safe = (θ′, ~6θ′)
        low, high = (1e4, theta_new) if zone == "flip" else (theta_new * 1.01, theta_new * 6)
    else:  # ≥ rule: flip = [θ, θ′), safe = [θ′, ~2θ′)
        low, high = (theta, theta_new * 0.99) if zone == "flip" else (theta_new * 1.01, theta_new * 2)
    if is_amount:
        values = np.exp(rng.uniform(np.log(low), np.log(high), n)).round()
        return values.astype("int64")
    return rng.uniform(low, high, n).round(2)


# 통과값 랜덤 컨텍스트 (context_mode="random_pass"): 실제 데이터의 문맥 다양성에 근접.
# locate에서는 모델 정답 필터 때문에 부적합했지만, 편집 요청은 모델이 문장을 맞힐 필요가
# 없으므로(목표 행동을 덮어쓰는 것) 사용 가능 — 합성→실제 key 전이 개선용.
RANDOM_PASS_RANGES = {
    "부채비율(%)": (10.0, 195.0),
    "유동비율(%)": (105.0, 400.0),
    "자기자본비율(%)": (25.0, 95.0),
    "자본총계": (1e10, 1.0e13),
    "보통주자본비율(%)": (5.0, 24.0),
    "기본자본비율(%)": (6.5, 24.0),
    "총자본비율(%)": (8.5, 24.0),
    "단순기본자본비율(%)": (3.2, 12.0),
    "공제후자본양수": (5e11, 3.7e13),
}


def sample_context_random_pass(rng, exclude_column, size):
    context = {}
    for column, (low, high) in RANDOM_PASS_RANGES.items():
        if column == exclude_column:
            continue
        values = rng.uniform(low, high, size)
        context[column] = (
            values.round().astype("int64") if column in AMOUNT_COLUMNS else values.round(2)
        )
    context["총자산순이익률ROA(%)"] = rng.normal(0.3, 0.5, size).round(2)
    context["영업흑자"] = rng.integers(50_000, 1_500_000, size)
    context["이익잉여금양수"] = rng.integers(50_000, 1_500_000, size)
    context["예대율(%)"] = np.clip(rng.normal(100, 7.5, size), 70, 130).round(2)
    return context


def build_rows(case, values, rng, context_mode="fixed_partner"):
    """값 배열 → 13개 지표 행. context_mode: fixed_partner(학습 재현) | random_pass(다양화)."""
    column = CASE_SPECS[case]["column"]
    if context_mode == "random_pass":
        context = sample_context_random_pass(rng, column, len(values))
    else:
        context = sample_context(rng, column, len(values))
        context.pop("_partner")
    rows = pd.DataFrame(context)
    rows[column] = values
    return rows[FEATURE_COLUMNS]


def render_rows(tokenizer, rows, column, max_length=256):
    """행 → (문장, value_last 토큰 위치, 전체 input_ids)."""
    texts, positions = [], []
    for _, row in rows.iterrows():
        text, spans = build_prompt_with_spans(row)
        info = build_positions(tokenizer, text, spans, column, max_length)
        texts.append(text)
        positions.append(info["value_last"][-1])
    return texts, positions


def _pad_batch(tokenizer, texts, device, max_length=256):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def collect_keys_values(model, tokenizer, texts, positions, layer, batch_size=32):
    """대상 층에서 (k = GELU(fc1) 출력, v = fc2 출력)을 value_last 위치에서 수집."""
    device = next(model.parameters()).device
    block = model.roberta.encoder.layer[layer]
    keys, values = [], []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        pos = torch.tensor(positions[start : start + batch_size], device=device)
        cache = {}
        h1 = block.intermediate.register_forward_hook(lambda m, i, o: cache.__setitem__("k", o))
        h2 = block.output.dense.register_forward_hook(lambda m, i, o: cache.__setitem__("v", o))
        try:
            model(**_pad_batch(tokenizer, chunk, device))
        finally:
            h1.remove(); h2.remove()
        idx = torch.arange(len(chunk), device=device)
        keys.append(cache["k"][idx, pos].float().cpu())
        values.append(cache["v"][idx, pos].float().cpu())
    return torch.cat(keys), torch.cat(values)  # [n, 4096], [n, 1024]


@torch.no_grad()
def compute_covariance(model, tokenizer, texts, layer, batch_size=32):
    """일반 프롬프트의 전(비패딩) 위치 key로 C = E[kkᵀ] 추정."""
    device = next(model.parameters()).device
    block = model.roberta.encoder.layer[layer]
    C = torch.zeros(4096, 4096, dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(texts), batch_size):
        batch = _pad_batch(tokenizer, texts[start : start + batch_size], device)
        cache = {}
        h = block.intermediate.register_forward_hook(lambda m, i, o: cache.__setitem__("k", o))
        try:
            model(**batch)
        finally:
            h.remove()
        k = cache["k"][batch["attention_mask"].bool()].double()  # [n_tokens, 4096]
        C += k.T @ k
        count += k.shape[0]
    return (C / count).float().cpu()


def optimize_v_star(model, tokenizer, texts, positions, layer, target_label,
                    steps=30, lr=0.3, clamp_factor=4.0, batch_size=32):
    """fc2 출력을 대체할 v*를 요청별로 탐색: 분류가 target_label이 되도록."""
    device = next(model.parameters()).device
    block = model.roberta.encoder.layer[layer]
    _, v0 = collect_keys_values(model, tokenizer, texts, positions, layer)
    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        pos = torch.tensor(positions[start : start + batch_size], device=device)
        batch = _pad_batch(tokenizer, chunk, device)
        v_init = v0[start : start + batch_size].to(device)
        v = v_init.clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        idx = torch.arange(len(chunk), device=device)
        target = torch.full((len(chunk),), target_label, device=device, dtype=torch.long)
        max_norm = v_init.norm(dim=-1, keepdim=True) * clamp_factor

        def hook(m, i, o):
            o = o.clone()
            o[idx, pos] = v.to(o.dtype)
            return o

        handle = block.output.dense.register_forward_hook(hook)
        try:
            for _ in range(steps):
                opt.zero_grad()
                out = model(**batch)
                loss = torch.nn.functional.cross_entropy(out.logits.float(), target)
                loss += 1e-3 * ((v - v_init) ** 2).sum(-1).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():  # 노름 폭주 방지
                    norm = v.norm(dim=-1, keepdim=True)
                    v.copy_(torch.where(norm > max_norm, v * max_norm / norm, v))
        finally:
            handle.remove()
        results.append(v.detach().float().cpu())
    return torch.cat(results), v0  # v* [R,1024], 원래 v [R,1024]


def build_edit_inputs(case, tokenizer, rng, n_requests=64, n_general=512, n_preserve=1024,
                      context_mode="fixed_partner"):
    """편집 요청·C용 일반·K₀용 보존 프롬프트 일괄 생성 (전부 합성)."""
    column = CASE_SPECS[case]["column"]
    req_rows = build_rows(
        case, sample_zone_values(case, "flip", n_requests, rng), rng, context_mode
    )
    req_texts, req_pos = render_rows(tokenizer, req_rows, column)

    # 일반 프롬프트: 9개 rule 각각의 경계 주변 양측 혼합 (C 추정용)
    gen_texts = []
    per_rule = max(1, n_general // 9)
    for c in CASE_SPECS:
        col_c = CASE_SPECS[c]["column"]
        vals = np.concatenate([
            sample_zone_values(c, "flip", per_rule // 2 + 1, rng),
            sample_zone_values(c, "safe", per_rule // 2 + 1, rng),
        ])[:per_rule]
        rows = build_rows(c, vals, rng, context_mode)
        gen_texts += [build_prompt_with_spans(r)[0] for _, r in rows.iterrows()]

    # 보존 프롬프트: target rule 안전존 + 다른 rule 문맥 (value_last 위치 key = K₀)
    pres_rows = build_rows(case, sample_zone_values(case, "safe", n_preserve // 2, rng), rng, context_mode)
    pres_texts, pres_pos = render_rows(tokenizer, pres_rows, column)
    for c in CASE_SPECS:
        if c == case:
            continue
        col_c = CASE_SPECS[c]["column"]
        vals = sample_zone_values(c, "safe", n_preserve // 16, rng)
        rows = build_rows(c, vals, rng, context_mode)
        t, p = render_rows(tokenizer, rows, col_c)
        pres_texts += t
        pres_pos += p

    return {
        "requests": (req_texts, req_pos, req_rows),
        "general": gen_texts,
        "preserve": (pres_texts, pres_pos),
    }
