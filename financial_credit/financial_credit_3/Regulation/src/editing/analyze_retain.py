"""편집 런들의 retain 오류를 rule별로 분해 (교차 오염 분석).

각 (case, method) 런에 대해 w2_delta를 재적용해 test.csv 예측을 복원하고
(predictions.csv로 저장), retain 오류를 다음으로 귀속한다:

- lost:{rule j}   : 라벨 0(rule j 하나만 탈락)인 행을 통과로 오판 — rule j 판정 상실
- over:{rule j}   : 라벨 1(전부 통과)인 행을 탈락으로 오판 — 경계까지의 정규화 마진이
                    가장 작은 rule j로 귀속 (경계 과잉 이동/오염의 근사 지표)
- multi_fail      : 여러 rule이 동시 탈락인 행의 오판 (귀속 불가)

출력: result/edit_main/{method}/{run}/retain_breakdown.json + 종합 retain_breakdown.md
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

from src.editing.kv import CASE_SPECS
from src.editing.generate_locate_pairs import RULE_SPECS
from src.models.test import format_prompt

RULE_OPS = {  # (column, 통과 판정 함수 팩토리) — check_rule_labels.RULES와 동일 의미
    "부채비율(%)": lambda th: lambda v: v <= th,
    "유동비율(%)": lambda th: lambda v: v >= th,
    "자기자본비율(%)": lambda th: lambda v: v >= th,
    "자본총계": lambda th: lambda v: v > th,
    "보통주자본비율(%)": lambda th: lambda v: v >= th,
    "기본자본비율(%)": lambda th: lambda v: v >= th,
    "총자본비율(%)": lambda th: lambda v: v >= th,
    "단순기본자본비율(%)": lambda th: lambda v: v >= th,
    "공제후자본양수": lambda th: lambda v: v > th,
}
SIGMA = {col: sigma for _, col, _, sigma, *_ in [(c, col, th, sg, ps, dl, ia) for (c, col, th, sg, ps, dl, ia) in RULE_SPECS]}
THETA = {col: th for (_, col, th, *_ ) in RULE_SPECS}


def case_rule_thresholds(case):
    th = dict(THETA)
    th[CASE_SPECS[case]["column"]] = CASE_SPECS[case]["theta_new"]
    return th


def pass_matrix(df, thresholds):
    cols = list(RULE_OPS)
    mat = np.zeros((len(df), len(cols)), dtype=bool)
    for j, col in enumerate(cols):
        v = pd.to_numeric(df[col], errors="coerce")
        mat[:, j] = RULE_OPS[col](thresholds[col])(v).fillna(False).to_numpy()
    return mat, cols


@torch.no_grad()
def predict(model, tokenizer, texts, device, batch_size=256):
    preds = []
    for s in tqdm(range(0, len(texts), batch_size), desc="Predict", leave=False):
        enc = tokenizer(texts[s : s + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def breakdown(case, preds, base, edited):
    thresholds = case_rule_thresholds(case)
    mat, cols = pass_matrix(base, thresholds)
    label_new = edited["label"].astype(int).to_numpy()
    flip = base["label"].astype(int).to_numpy() != label_new
    retain, err = ~flip, (preds != label_new)
    idx = np.flatnonzero(retain & err)

    n_fail = (~mat).sum(1)
    # 라벨1 행의 정규화 마진 (경계까지 σ 단위 거리)
    margins = np.stack([
        np.abs(pd.to_numeric(base[col], errors="coerce").to_numpy() - thresholds[col]) / SIGMA[col]
        for col in cols
    ], axis=1)

    out = {}
    for i in idx:
        if label_new[i] == 0:
            key = f"lost:{cols[np.flatnonzero(~mat[i])[0]]}" if n_fail[i] == 1 else "multi_fail"
        else:
            key = f"over:{cols[int(np.nanargmin(margins[i]))]}"
        out[key] = out.get(key, 0) + 1
    return {"n_retain_err": int(len(idx)), "n_retain": int(retain.sum()), "breakdown": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/edit_main")
    parser.add_argument("--model_name", default="yongukpark/xlm-roberta-large-rule")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = pd.read_csv("data/test/text/test.csv", low_memory=False).dropna()
    texts = base.apply(format_prompt, axis=1).tolist()
    edited_all = {
        c: pd.read_csv(f"data/test/text/edited_rule_25pct/test_case{c}.csv", low_memory=False).dropna()
        for c in range(1, 10)
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    results = {}
    for run_dir in sorted(Path(args.root).glob("*/case*")):
        meta_p = run_dir / "metrics.json"
        delta_p = run_dir / "w2_delta.pt"
        if not (meta_p.exists() and delta_p.exists()):
            continue
        meta = json.load(open(meta_p))
        case, method = meta["case"], meta["method"]
        deltas = torch.load(delta_p, map_location=device)

        applied = []
        for l, d in deltas.items():  # 델타 적용
            w = model.roberta.encoder.layer[int(l)].output.dense.weight
            d = d.to(w.dtype).to(device)
            w.data += d
            applied.append((w, d))
        try:
            preds = predict(model, tokenizer, texts, device)
        finally:
            for w, d in applied:  # 원상 복구
                w.data -= d

        pd.DataFrame({"index": base.index, "pred": preds}).to_csv(run_dir / "predictions.csv", index=False)
        res = breakdown(case, preds, base, edited_all[case])
        with (run_dir / "retain_breakdown.json").open("w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        results[(case, method)] = res
        top = sorted(res["breakdown"].items(), key=lambda kv: -kv[1])[:3]
        print(f"case{case} {method:10s} 오류 {res['n_retain_err']:>6,}/{res['n_retain']:,} | {top}")

    # 종합 md
    lines = ["# Retain 오류 rule별 분해 (교차 오염 분석)\n",
             "생성: `src/editing/analyze_retain.py` — 각 런의 retain 오류를 원인 rule로 귀속.",
             "`lost:X` = rule X 판정 상실(라벨0→통과 오판) / `over:X` = X 경계 근처 과잉 거절(라벨1→탈락 오판)\n"]
    for (case, method), res in sorted(results.items()):
        col = CASE_SPECS[case]["column"]
        lines.append(f"## case {case} ({col}) — {method}: 오류 {res['n_retain_err']:,}/{res['n_retain']:,}")
        for k, v in sorted(res["breakdown"].items(), key=lambda kv: -kv[1]):
            mark = " ← **target rule**" if k.endswith(col) else ""
            lines.append(f"- {k}: {v:,} ({v/max(res['n_retain_err'],1)*100:.0f}%){mark}")
        lines.append("")
    Path(args.root, "retain_breakdown.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {args.root}/retain_breakdown.md")


if __name__ == "__main__":
    main()
