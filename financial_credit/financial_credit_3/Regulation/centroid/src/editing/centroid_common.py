"""EDRM centroid 기반 locate-then-edit 실험 공통 유틸 (Exp1~5 공유).

기존 인프라를 그대로 재사용한다 (새로 만들지 않음):
  - src.dataset.check_rule_labels: RULES, OPS, compute_rule_labels
  - src.editing.generate_locate_pairs: FEATURE_COLUMNS, RULE_SPECS, CONTEXT_FILL, AMOUNT_COLUMNS
  - src.editing.locate: build_prompt_with_spans, build_positions, char_span_to_token_span,
    find_name_char_span, value_to_str  (offset-mapping 기반 토큰 span 탐지 — 실측 검증된 방식)
  - src.models.test: format_prompt, format_integer_like

이 실험은 rule 1 (부채비율(%) 200 -> 85)에 초점을 둔다. RULE_SPECS를 통해 다른 rule로도
확장 가능하도록 rule 번호를 인자로 받는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 이 파일 위치: Regulation/centroid/src/editing/centroid_common.py
CENTROID_ROOT = Path(__file__).resolve().parents[2]  # Regulation/centroid/
REGULATION_ROOT = CENTROID_ROOT.parent  # Regulation/ -- src.dataset/src.editing/src.models 재사용에 필요
PROJECT_ROOT = REGULATION_ROOT.parent  # repo root -- git 커밋 해시 조회용
if str(REGULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGULATION_ROOT))

from src.dataset.check_rule_labels import RULES, OPS, compute_rule_labels  # noqa: E402
from src.editing.generate_locate_pairs import FEATURE_COLUMNS, RULE_SPECS  # noqa: E402
from src.editing.locate import build_prompt_with_spans, build_positions  # noqa: E402
from src.models.test import format_prompt  # noqa: E402

# 로컬에 미리 받아둔 모델 사본 (yongukpark/xlm-roberta-large-rule과 동일, 재다운로드 회피)
LOCAL_MODEL_DIR = "/data1/jinsil/aidep/editing_centroid/xlm-roberta-large-rule"
HF_MODEL_NAME = "yongukpark/xlm-roberta-large-rule"

TEST_CSV = REGULATION_ROOT / "data" / "test" / "text" / "test.csv"
EDITED_CASE_DIR = REGULATION_ROOT / "data" / "test" / "text" / "edited_rule_25pct"
RESULT_ROOT = CENTROID_ROOT / "result" / "centroid"
CACHE_DIR = RESULT_ROOT / "cache"

NUM_LAYERS = 24
N_HIDDEN_STATES = NUM_LAYERS + 1  # hidden_states 길이: 0=embedding, 1..24=layer 출력
MAX_LEN = 256

RULE_SPEC_BY_NUM = {s[0]: s for s in RULE_SPECS}
RULE_COLUMN_BY_NUM = {s[0]: s[1] for s in RULE_SPECS}


def rule_spec(rule_num: int):
    """(case, column, theta, sigma, pass_side, delta, is_amount)"""
    return RULE_SPEC_BY_NUM[rule_num]


def other_rules_pass_mask(df: pd.DataFrame, target_column: str) -> pd.Series:
    """target_column을 제외한 나머지 8개 rule을 모두 통과하는 행 마스크 (decisive set 정의)."""
    passed = pd.Series(True, index=df.index)
    for column, op, threshold in RULES:
        if column == target_column:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        passed &= OPS[op](values, threshold).fillna(False)
    return passed


def load_test_csv() -> pd.DataFrame:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"{TEST_CSV} 없음. 먼저 `python src/dataset/generate_test_csv.py` 실행 필요."
        )
    return pd.read_csv(TEST_CSV, low_memory=False).dropna().reset_index(drop=True)


def decisive_split(df: pd.DataFrame, rule_num: int, n_per_group=500, seed=0):
    """rule_num의 decisive set을 현 threshold 기준으로 A(통과)/B(탈락) 분할, 그룹당 최대 n_per_group."""
    _, column, theta, _, pass_side, _, _ = rule_spec(rule_num)
    decisive = other_rules_pass_mask(df, column)
    values = pd.to_numeric(df[column], errors="coerce")
    if pass_side == "below":
        a_mask = decisive & (values <= theta)
        b_mask = decisive & (values > theta)
    else:
        a_mask = decisive & (values >= theta)
        b_mask = decisive & (values < theta)
    rng = np.random.default_rng(seed)
    a_idx = df.index[a_mask]
    b_idx = df.index[b_mask]
    a_sel = rng.choice(a_idx, size=min(n_per_group, len(a_idx)), replace=False)
    b_sel = rng.choice(b_idx, size=min(n_per_group, len(b_idx)), replace=False)
    return df.loc[a_sel].reset_index(drop=True), df.loc[b_sel].reset_index(drop=True)


# --------------------------------------------------------------------------
# 모델 로딩 / 추론
# --------------------------------------------------------------------------
def load_model(model_name=LOCAL_MODEL_DIR, device="cuda:0", dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(model, tokenizer, texts, device, batch_size=128, max_len=MAX_LEN):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True,
                         return_tensors="pt").to(device)
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def row_positions(tokenizer, row: pd.Series, rule_column: str, max_length=MAX_LEN):
    """format_prompt와 동일한 문장 + 토큰 위치(dict: value_last/value_all/name/cls/q_last/ctrl_roa)."""
    feat_row = {c: row[c] for c in FEATURE_COLUMNS}
    text, spans = build_prompt_with_spans(feat_row)
    info = build_positions(tokenizer, text, spans, rule_column, max_length)
    return text, info


@torch.no_grad()
def extract_layer_position_hidden(model, tokenizer, df: pd.DataFrame, rule_column: str, device,
                                   batch_size=32, max_len=MAX_LEN):
    """
    각 레이어(0..24) x 위치(CLS, rule 값 span 평균) hidden state 추출.
    반환: {"cls": (N,25,H), "debt_span": (N,25,H)}  (dict 키 이름은 범용화해 rule_column 무관하게 고정)
    """
    texts, infos = [], []
    for _, row in df.iterrows():
        text, info = row_positions(tokenizer, row, rule_column, max_len)
        texts.append(text)
        infos.append(info)

    cls_out, span_out = [], []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_infos = infos[i:i + batch_size]
        ids = [info["input_ids"] for info in batch_infos]
        enc = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)  # (B, 25, T, H)
        cls_out.append(hs[:, :, 0, :].cpu())
        for b, info in enumerate(batch_infos):
            pos = info["value_all"]
            span_out.append(hs[b, :, pos, :].mean(dim=1).cpu().unsqueeze(0))
    cls_tensor = torch.cat(cls_out, dim=0)
    span_tensor = torch.cat(span_out, dim=0)
    return {"cls": cls_tensor, "debt_span": span_tensor}, texts, infos


def get_cached_hidden(cache_path: Path, model, tokenizer, df: pd.DataFrame, rule_column: str, device,
                       batch_size=32):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        d = torch.load(cache_path)
        return d["cls"], d["debt_span"]
    hid, _, _ = extract_layer_position_hidden(model, tokenizer, df, rule_column, device, batch_size)
    torch.save(hid, cache_path)
    return hid["cls"], hid["debt_span"]


def ensure_dirs():
    for d in (RESULT_ROOT, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# FFN intermediate(4096차원, gelu 후) 추출 — rank-1 weight edit(Exp4b)의 key 공간.
# output.dense(fc2)의 입력이 이 값이므로, ROME류 rank-1 edit은 여기를 key로 쓴다.
# --------------------------------------------------------------------------
@torch.no_grad()
def extract_ffn_intermediate_debt_span(model, tokenizer, df: pd.DataFrame, rule_column: str,
                                        encoder_layer_idx: int, device, batch_size=32, max_len=MAX_LEN):
    """model.roberta.encoder.layer[encoder_layer_idx].intermediate 출력(gelu 후, 4096차원)을
    debt_span 토큰 위치에서 평균 pooling. hidden_states 인덱스 L = encoder_layer_idx + 1."""
    texts, infos = [], []
    for _, row in df.iterrows():
        text, info = row_positions(tokenizer, row, rule_column, max_len)
        texts.append(text)
        infos.append(info)

    captured = {}

    def hook(module, inp, out):
        captured["val"] = out.detach()

    module = model.roberta.encoder.layer[encoder_layer_idx].intermediate
    handle = module.register_forward_hook(hook)
    outs = []
    try:
        for i in range(0, len(texts), batch_size):
            batch_infos = infos[i:i + batch_size]
            ids = [info["input_ids"] for info in batch_infos]
            enc = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)
            ffn = captured["val"]  # (B, T, 4096)
            for b, info in enumerate(batch_infos):
                pos = info["value_all"]
                outs.append(ffn[b, pos, :].mean(dim=0).cpu().unsqueeze(0))
    finally:
        handle.remove()
    return torch.cat(outs, dim=0)


def get_cached_ffn_intermediate(cache_path: Path, model, tokenizer, df: pd.DataFrame, rule_column: str,
                                 encoder_layer_idx: int, device, batch_size=32):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return torch.load(cache_path)
    x = extract_ffn_intermediate_debt_span(model, tokenizer, df, rule_column, encoder_layer_idx, device,
                                            batch_size)
    torch.save(x, cache_path)
    return x


def other_rule_fail_mask(df: pd.DataFrame, rule_num: int) -> pd.Series:
    """target rule(rule_num) 자체는 통과(구 threshold 기준)하지만 label=0인 행
    -> 다른 rule(2~9 등) 때문에 탈락한 실데이터 ("타 rule 탈락" 그룹, Exp4.0 key 검증용)."""
    _, column, theta, _, pass_side, _, _ = rule_spec(rule_num)
    values = pd.to_numeric(df[column], errors="coerce")
    self_pass = (values <= theta) if pass_side == "below" else (values >= theta)
    return self_pass & (df["label"] == 0)


def real_band_mask(df: pd.DataFrame, rule_num: int, new_threshold: float) -> pd.Series:
    """실데이터에서 '부채비율 외 8개 rule 통과' + 구 threshold는 통과, 신 threshold는 탈락하는
    밴드 샘플 (new_threshold < value <= theta_old)."""
    _, column, theta, _, pass_side, _, _ = rule_spec(rule_num)
    decisive = other_rules_pass_mask(df, column)
    values = pd.to_numeric(df[column], errors="coerce")
    if pass_side == "below":
        return decisive & (values > new_threshold) & (values <= theta)
    return decisive & (values < new_threshold) & (values >= theta)


# --------------------------------------------------------------------------
# run 위생: 실행마다 고유 디렉토리 + manifest + 덮어쓰기 가드.
# (단일층 rank-1 체크포인트가 이후 진단 실행에 덮어써져 추적이 안 됐던 사고 재발 방지용)
# --------------------------------------------------------------------------
import json as _json  # noqa: E402
from datetime import datetime as _datetime  # noqa: E402


def new_run_dir(base_dir: Path, method: str, case) -> Path:
    """base_dir/{method}_case{case}_{YYYYMMDD_HHMMSS}/ 를 만들어 반환. 실행마다 고유."""
    stamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{method}_case{case}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git_commit_hash():
    """현재 HEAD 커밋 해시 + dirty 여부. 커밋이 없으면 (None, None) -- 과거 manifest들이
    해시가 비어 있었던 이유(이 함수 자체가 존재하지 않았음 + 애초에 커밋도 없었음)."""
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                 text=True, timeout=5)
        if commit.returncode != 0:
            return None, None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=5)
        return commit.stdout.strip(), bool(dirty.stdout.strip())
    except Exception:
        return None, None


def write_manifest(run_dir: Path, script: str, inputs: dict, extra: dict = None, start_time=None):
    """run_dir/manifest.json 기록. start_time을 넘기면 종료 시각도 함께 기록(호출 시점 = 종료).
    git commit hash도 기록 -- 커밋이 없는 상태에서 쓰면 commit_hash=null, dirty=null이 된다
    (이 필드가 비어 있다면 "그 시점에 커밋이 없었다"는 뜻으로 해석하면 됨)."""
    commit_hash, dirty = _git_commit_hash()
    manifest = {
        "script": script,
        "inputs": inputs,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": _datetime.now().isoformat(),
        "git_commit_hash": commit_hash,
        "git_dirty": dirty,
    }
    if extra:
        manifest.update(extra)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        _json.dump(manifest, f, ensure_ascii=False, indent=2)


def assert_no_overwrite(path: Path, overwrite: bool = False):
    """path가 이미 존재하면 --overwrite 없이는 실패시켜 silent overwrite를 막는다."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} 이미 존재함. 덮어쓰려면 --overwrite 플래그를 명시적으로 지정하세요."
        )
