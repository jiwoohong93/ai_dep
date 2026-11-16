"""Causal tracing (activation patching)으로 rule별 반응 위치를 locate한다.

최소쌍(base=통과, source=탈락)의 hidden state를 (layer, position group)별로 교체해
분류 확률 변화(indirect effect)를 측정한다. 쌍별 IE를 저장한 뒤 누적 예산
(1k, 2k, ..., 10k)별 히트맵과 피크 층을 result/locate/rule{i}/n{k}/에 저장한다.

사용 예:
    python src/editing/locate.py --rule 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.editing.generate_locate_pairs import FEATURE_COLUMNS, RULE_SPECS
from src.models.test import format_integer_like

# format_prompt와 동일한 문장을 세그먼트 단위로 조립해 각 지표 값의 문자 위치를 추적한다.
# (아래 구조를 바꾸면 안 됨 — build_prompt_with_spans가 format_prompt와의 일치를 assert)
PROMPT_SEGMENTS = [
    ("어떤 은행의 보통주자본비율이 ", None),
    (None, "보통주자본비율(%)"),
    ("%, 기본자본비율이 ", None),
    (None, "기본자본비율(%)"),
    ("%, 총자본비율이 ", None),
    (None, "총자본비율(%)"),
    ("%, 단순기본자본비율이 ", None),
    (None, "단순기본자본비율(%)"),
    ("%, 공제후자본양수가 ", None),
    (None, "공제후자본양수"),
    (", 총자산순이익률ROA가 ", None),
    (None, "총자산순이익률ROA(%)"),
    ("%, 영업흑자가 ", None),
    (None, "영업흑자"),
    (", 이익잉여금양수가 ", None),
    (None, "이익잉여금양수"),
    (", 예대율이 ", None),
    (None, "예대율(%)"),
    ("%입니다. 어떤 기업의 부채비율이 ", None),
    (None, "부채비율(%)"),
    ("%, 유동비율이 ", None),
    (None, "유동비율(%)"),
    ("%, 자기자본비율이 ", None),
    (None, "자기자본비율(%)"),
    ("%이고, 자본총계가 ", None),
    (None, "자본총계"),
    ("입니다. 이 은행은 이 기업에게 대출할 수 있나요?", None),
]
AMOUNT_LIKE = {"공제후자본양수", "영업흑자", "이익잉여금양수", "자본총계"}
GROUPS = ["value_last", "value_all", "name", "cls", "q_last", "ctrl_roa"]


def value_to_str(column, value):
    if column in AMOUNT_LIKE:
        return format_integer_like(value)
    return str(value)


def build_prompt_with_spans(row):
    """format_prompt와 동일한 문장 + 지표별 값의 (char_start, char_end)."""
    parts, spans, cursor = [], {}, 0
    for literal, column in PROMPT_SEGMENTS:
        piece = literal if column is None else value_to_str(column, row[column])
        if column is not None:
            spans[column] = (cursor, cursor + len(piece))
        parts.append(piece)
        cursor += len(piece)
    return "".join(parts), spans


def char_span_to_token_span(offsets, char_start, char_end):
    positions = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if start < char_end and end > char_start and end > start
    ]
    return positions


def find_name_char_span(text, column, value_char_start):
    """지표명 라벨의 문자 구간 — 값 span 바로 앞에서 결정적으로 계산.

    검색(find/rfind) 금지: '기본자본비율'은 '단순기본자본비율'의 부분 문자열이라
    검색 기반은 다른 지표 라벨 내부를 잡을 수 있다. 값 앞은 항상 "{label}이 " 또는
    "{label}가 " 형태이므로 값 시작 위치에서 역산한다.
    """
    label = column.replace("(%)", "")
    for suffix in ("이 ", "가 "):
        start = value_char_start - len(suffix) - len(label)
        if start >= 0 and text[start : value_char_start] == label + suffix:
            return start, start + len(label)
    raise ValueError(f"label not found before value: {column}")


def build_positions(tokenizer, text, spans, rule_column, max_length=256):
    encoded = tokenizer(
        text, return_offsets_mapping=True, truncation=True, max_length=max_length
    )
    offsets = encoded["offset_mapping"]
    seq_len = len(encoded["input_ids"])

    value_positions = char_span_to_token_span(offsets, *spans[rule_column])
    name_positions = char_span_to_token_span(
        offsets, *find_name_char_span(text, rule_column, spans[rule_column][0])
    )
    ctrl_positions = char_span_to_token_span(offsets, *spans["총자산순이익률ROA(%)"])
    if not value_positions:
        raise ValueError("value token span is empty")
    return {
        "input_ids": encoded["input_ids"],
        "seq_len": seq_len,
        "value_last": [value_positions[-1]],
        "value_all": value_positions,
        "name": name_positions,
        "cls": [0],
        "q_last": [seq_len - 2],  # </s> 앞 마지막 내용 토큰
        "ctrl_roa": ctrl_positions,
    }


def align_positions(base_info, source_info, group):
    """base/source 토큰 위치 정렬.

    모든 그룹의 위치는 각자의 문장에서 독립 계산되므로(문자 span 기반) 같은 내용
    토큰끼리 짝을 이룬다. 값 span만 텍스트 자체가 달라 끝 토큰 기준으로 정렬한다.
    """
    base_pos, source_pos = base_info[group], source_info[group]
    n = min(len(base_pos), len(source_pos))
    if group in ("value_last", "value_all"):
        return base_pos[-n:], source_pos[-n:]  # 숫자 집계는 마지막 토큰에 수렴
    return base_pos[:n], source_pos[:n]


def parse_args():
    parser = argparse.ArgumentParser(description="Locate rule circuits via activation patching.")
    parser.add_argument("--rule", type=int, required=True, choices=range(1, 10))
    parser.add_argument("--pairs_dir", default="data/rules/edited_rule_25pct/locate")
    parser.add_argument("--output_dir", default="result/locate")
    parser.add_argument("--model_name", default="yongukpark/xlm-roberta-large-rule")
    parser.add_argument("--budgets", default="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_pairs", type=int, default=None, help="디버그용 상한")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.budgets = [int(b) for b in args.budgets.split(",")]
    return args


def encode_batch(tokenizer, infos, device, max_length):
    ids = [info["input_ids"] for info in infos]
    encoded = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def gather_patch_indices(base_infos, source_infos, group):
    """배치 전체의 (batch_idx, base_pos), (batch_idx, source_pos) 평탄화 인덱스."""
    batch_idx, base_flat, source_flat = [], [], []
    for i, (b, s) in enumerate(zip(base_infos, source_infos)):
        bp, sp = align_positions(b, s, group)
        n = min(len(bp), len(sp))
        batch_idx.extend([i] * n)
        base_flat.extend(bp[:n])
        source_flat.extend(sp[:n])
    return (
        torch.tensor(batch_idx, dtype=torch.long),
        torch.tensor(base_flat, dtype=torch.long),
        torch.tensor(source_flat, dtype=torch.long),
    )


@torch.inference_mode()
def run_clean(model, batch, output_hidden_states=False):
    out = model(**batch, output_hidden_states=output_hidden_states)
    probs = torch.softmax(out.logits.float(), dim=-1)
    hidden = out.hidden_states if output_hidden_states else None
    return probs, hidden


@torch.inference_mode()
def run_patched(model, layer_module, batch, donor_hidden, batch_idx, dst_pos, src_pos):
    def hook(module, inputs, output):
        h = output[0]
        h[batch_idx, dst_pos] = donor_hidden[batch_idx, src_pos].to(h.dtype)
        return (h,) + tuple(output[1:])

    handle = layer_module.register_forward_hook(hook)
    try:
        out = model(**batch)
    finally:
        handle.remove()
    return torch.softmax(out.logits.float(), dim=-1)


def main():
    args = parse_args()
    spec = next(s for s in RULE_SPECS if s[0] == args.rule)
    _, rule_column, *_ = spec

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    layers = model.roberta.encoder.layer
    num_layers = len(layers)

    pairs = pd.read_csv(Path(args.pairs_dir) / f"rule{args.rule}.csv", low_memory=False)
    if args.max_pairs:
        pairs = pairs.head(args.max_pairs)
    print(f"rule {args.rule} ({rule_column}): {len(pairs):,}쌍, layers={num_layers}")

    # 문장 렌더 + 값 토큰 span (base/source)
    base_infos, source_infos = [], []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Building spans"):
        base_row = {c: row[c] for c in FEATURE_COLUMNS}
        source_row = dict(base_row)
        source_row[rule_column] = (
            int(row["value_fail"]) if rule_column in AMOUNT_LIKE else row["value_fail"]
        )
        for target, store in ((base_row, base_infos), (source_row, source_infos)):
            text, spans = build_prompt_with_spans(target)
            store.append(build_positions(tokenizer, text, spans, rule_column, args.max_length))

    n_pairs = len(pairs)
    ie = {
        "p2f": np.zeros((n_pairs, num_layers, len(GROUPS)), dtype=np.float16),
        "f2p": np.zeros((n_pairs, num_layers, len(GROUPS)), dtype=np.float16),
    }
    valid = np.zeros(n_pairs, dtype=bool)

    for start in tqdm(range(0, n_pairs, args.batch_size), desc="Tracing"):
        idx = slice(start, min(start + args.batch_size, n_pairs))
        b_infos = base_infos[idx]
        s_infos = source_infos[idx]
        base_batch = encode_batch(tokenizer, b_infos, args.device, args.max_length)
        source_batch = encode_batch(tokenizer, s_infos, args.device, args.max_length)

        base_probs, base_hidden = run_clean(model, base_batch, output_hidden_states=True)
        source_probs, source_hidden = run_clean(model, source_batch, output_hidden_states=True)

        # 모델 정답 필터: base→1, source→0 예측 쌍만 유효
        ok = (base_probs.argmax(-1) == 1) & (source_probs.argmax(-1) == 0)
        valid[idx] = ok.cpu().numpy()

        for g, group in enumerate(GROUPS):
            batch_idx, dst_pos, src_pos = gather_patch_indices(b_infos, s_infos, group)
            batch_idx, dst_pos, src_pos = (
                batch_idx.to(args.device),
                dst_pos.to(args.device),
                src_pos.to(args.device),
            )
            for layer in range(num_layers):
                # p2f: base 문장에 source 상태 주입 → P(0) 상승분
                probs = run_patched(
                    model, layers[layer], base_batch,
                    source_hidden[layer + 1], batch_idx, dst_pos, src_pos,
                )
                ie["p2f"][idx, layer, g] = (
                    (probs[:, 0] - base_probs[:, 0]).cpu().numpy().astype(np.float16)
                )
                # f2p: source 문장에 base 상태 주입 → P(1) 상승분
                probs = run_patched(
                    model, layers[layer], source_batch,
                    base_hidden[layer + 1], batch_idx, src_pos, dst_pos,
                )
                ie["f2p"][idx, layer, g] = (
                    (probs[:, 1] - source_probs[:, 1]).cpu().numpy().astype(np.float16)
                )

    out_root = Path(args.output_dir) / f"rule{args.rule}"
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_root / "pair_ie.npz",
        p2f=ie["p2f"], f2p=ie["f2p"], valid=valid, groups=np.array(GROUPS),
    )
    print(f"유효쌍: {valid.sum():,}/{n_pairs:,} ({valid.mean()*100:.1f}%)")

    # 누적 예산별 집계
    for budget in args.budgets:
        mask = valid.copy()
        mask[budget:] = False
        budget_dir = out_root / f"n{budget}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "rule": args.rule, "column": rule_column,
            "budget": budget, "n_valid_used": int(mask.sum()),
        }
        for direction in ("p2f", "f2p"):
            heatmap = ie[direction][mask].astype(np.float32).mean(axis=0)
            df = pd.DataFrame(
                heatmap, columns=GROUPS,
                index=[f"layer_{l}" for l in range(num_layers)],
            )
            df.to_csv(budget_dir / f"heatmap_{direction}.csv")
            peak = int(np.abs(heatmap[:, GROUPS.index("value_last")]).argmax())
            meta[f"peak_layer_{direction}"] = peak
            meta[f"memit_window_{direction}"] = [
                max(0, peak - 2), min(num_layers - 1, peak + 2),
            ]
        with (budget_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"n{budget}: valid {meta['n_valid_used']:,} | "
              f"peak p2f=L{meta['peak_layer_p2f']} f2p=L{meta['peak_layer_f2p']}")


if __name__ == "__main__":
    main()
