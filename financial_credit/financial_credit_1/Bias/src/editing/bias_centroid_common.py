"""Bias(UCI Adult) 도메인 centroid 실험 공용 유틸.

Regulation의 centroid_common.py/locate.py와 같은 철학 -- 다만 대상이:
- 모델: Bias/model/finetuned_xlm-roberta-large_cgmix_rel/best (won 학습, acc=0.8734 확인됨)
- 데이터: Bias/data/{train,test}/text/*.csv (test.csv는 최종 평가에만 사용 -- Regulation과 동일 규율)
- 편집 대상: "비혼 페널티 완화" (README 확정 목표) -- capital-gain(rule) 신호는 안 건드리면서
  비혼 표지(marital-status/relationship) 신호만 완화하는 위치를 찾아야 하므로, locate 단계부터
  두 신호(marital / cg)를 동시에 측정한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 이 파일 위치: Bias/src/editing/bias_centroid_common.py
BIAS_ROOT = Path(__file__).resolve().parents[2]  # Bias/
PROJECT_ROOT = BIAS_ROOT.parent  # repo root -- src.editing.centroid_edit_keyselect 등 재사용에 필요
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = BIAS_ROOT / "model" / "finetuned_xlm-roberta-large_cgmix_rel" / "best"
TRAIN_CSV = BIAS_ROOT / "data" / "train" / "text" / "train.csv"
TEST_CSV = BIAS_ROOT / "data" / "test" / "text" / "test.csv"  # 평가 전용 -- locate/gate/delta_v에 쓰지 않음
RESULT_ROOT = BIAS_ROOT / "result" / "bias_centroid"

N_HIDDEN_STATES = 25  # embedding(0) + encoder layer 1..24 출력
MAX_LEN = 256
CG_THETA = 7000.0  # capital-gain soft rule 임계값 (Bias/src/dataset/generate_cg_boundary_test.py와 동일)
CG_SIGMA = 0.15 * CG_THETA / 2  # = 525, 같은 파일의 관행

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country",
]
NUMERIC_COLUMNS = {"age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"}

# format_prompt(Bias/src/common/prompt.py)와 동일한 문장을 세그먼트 단위로 조립 --
# 구조를 바꾸면 안 됨(build_prompt_with_spans가 원본과 문자 단위로 일치해야 span이 맞음).
PROMPT_SEGMENTS = [
    ("A person's age is ", None),
    (None, "age"),
    (", sex is ", None),
    (None, "sex"),
    (", race is ", None),
    (None, "race"),
    (", native country is ", None),
    (None, "native-country"),
    (", marital status is ", None),
    (None, "marital-status"),
    (", and household relationship is ", None),
    (None, "relationship"),
    (". Their employment type is ", None),
    (None, "workclass"),
    (", occupation is ", None),
    (None, "occupation"),
    (", education level is ", None),
    (None, "education"),
    (", and years of education is ", None),
    (None, "education-num"),
    (". They work ", None),
    (None, "hours-per-week"),
    (" hours per week, with a capital gain of ", None),
    (None, "capital-gain"),
    (" dollars, a capital loss of ", None),
    (None, "capital-loss"),
    (" dollars, and a census sampling weight of ", None),
    (None, "fnlwgt"),
    (". Does this person's annual income exceed 50,000 dollars?", None),
]


def format_integer_like(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def value_to_str(column, value):
    if column in NUMERIC_COLUMNS:
        return format_integer_like(value)
    return str(value)


def build_prompt_with_spans(row):
    parts, spans, cursor = [], {}, 0
    for literal, column in PROMPT_SEGMENTS:
        piece = literal if column is None else value_to_str(column, row[column])
        if column is not None:
            spans[column] = (cursor, cursor + len(piece))
        parts.append(piece)
        cursor += len(piece)
    return "".join(parts), spans


def _assert_matches_reference(row):
    """Bias/src/common/prompt.py의 format_prompt와 문자 단위로 일치하는지 1회 검증용."""
    from Bias.src.common.prompt import format_prompt
    text, _ = build_prompt_with_spans(row)
    ref = format_prompt(row)
    assert text == ref, f"세그먼트 조립 결과가 원본 format_prompt와 다름:\n{text!r}\n{ref!r}"


def char_span_to_token_span(offsets, char_start, char_end):
    return [idx for idx, (s, e) in enumerate(offsets) if s < char_end and e > char_start and e > s]


def build_positions(tokenizer, text, spans, max_length=MAX_LEN):
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = encoded["offset_mapping"]
    seq_len = len(encoded["input_ids"])
    out = {"input_ids": encoded["input_ids"], "seq_len": seq_len, "cls": [0]}
    for col in ("marital-status", "relationship", "capital-gain", "sex"):
        pos = char_span_to_token_span(offsets, *spans[col])
        out[col] = pos
    return out


def row_positions(tokenizer, row: pd.Series, max_length=MAX_LEN):
    text, spans = build_prompt_with_spans(row)
    info = build_positions(tokenizer, text, spans, max_length)
    return text, info


def load_model(model_dir=MODEL_DIR, device="cuda:0", dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_texts(model, tokenizer, texts, device, batch_size=128, max_len=MAX_LEN):
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_len,
                         padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)


@torch.no_grad()
def extract_multi_position_hidden(model, tokenizer, df: pd.DataFrame, device, batch_size=32,
                                   max_len=MAX_LEN, positions=("cls", "marital-status", "relationship", "capital-gain")):
    """각 layer(0..24) x 지정 position들의 평균 hidden state. 반환: {pos_name: (N,25,H)}"""
    texts, infos = [], []
    for _, row in df.iterrows():
        text, info = row_positions(tokenizer, row, max_len)
        texts.append(text)
        infos.append(info)

    out = {p: [] for p in positions}
    for i in range(0, len(texts), batch_size):
        batch_infos = infos[i:i + batch_size]
        ids = [info["input_ids"] for info in batch_infos]
        enc = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)  # (B,25,T,H)
        for b, info in enumerate(batch_infos):
            for p in positions:
                pos_idx = info[p]
                out[p].append(hs[b, :, pos_idx, :].mean(dim=1).cpu().unsqueeze(0))
    return {p: torch.cat(v, dim=0) for p, v in out.items()}, texts, infos


def load_train_csv():
    return pd.read_csv(TRAIN_CSV).dropna().reset_index(drop=True)


def load_test_csv():
    """평가 전용 -- locate/delta_v/gate 구성에 사용 금지."""
    return pd.read_csv(TEST_CSV).dropna().reset_index(drop=True)


def fairness_metrics(df: pd.DataFrame, preds: np.ndarray):
    """Bias/src/models/test.py의 fairness_metrics와 동일 정의(deo_repo=acc gap, deo_lit=TPR gap)."""
    y = df["label"].to_numpy()
    male = (df["sex"] == "Male").to_numpy()
    if male.all() or (~male).all() or (y[male] == 1).sum() == 0 or (y[~male] == 1).sum() == 0:
        return {}
    acc_m = float((preds[male] == y[male]).mean())
    acc_f = float((preds[~male] == y[~male]).mean())
    tpr_m = float((preds[male & (y == 1)] == 1).mean())
    tpr_f = float((preds[~male & (y == 1)] == 1).mean())
    return {"acc_male": acc_m, "acc_female": acc_f, "deo_repo": abs(acc_m - acc_f),
            "tpr_male": tpr_m, "tpr_female": tpr_f, "deo_lit": abs(tpr_m - tpr_f)}


def new_run_dir(base_dir: Path, tag: str):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base_dir / f"{tag}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(run_dir: Path, script: str, inputs: dict, extra: dict, start_time):
    import json
    import subprocess
    from datetime import datetime
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                     capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    manifest = {
        "script": script, "inputs": {k: str(v) for k, v in inputs.items()},
        "start_time": str(start_time), "end_time": str(datetime.now()),
        "git_commit_hash": commit, "git_dirty": dirty, **(extra or {}),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
