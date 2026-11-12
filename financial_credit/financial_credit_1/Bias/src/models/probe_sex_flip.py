"""성별 반사실(sex flip) 진단 — DEO 편집 개입의 상한 추정.

test.csv 전 행에 대해 sex만 Male<->Female로 뒤집은 프롬프트를 만들어 예측을 비교:

- flip_rate            : sex 토큰 하나로 예측이 바뀌는 행 비율 (직접 경로의 세기)
- flip_rate_y1         : 실제 y=1 행에서의 비율 (TPR 격차 = DEO_lit과 직결)
- direction            : M->F로 바꿨을 때 1->0이 되는 비율 vs F->M로 바꿨을 때 0->1
- deo_lit_counterfactual: 모델이 완전 반사실-불변이라 가정하고 (원본과 뒤집기 예측을
  평균 0.5 반올림 대신, 두 예측이 같을 때만 채택 + 다르면 남성판 예측 사용 같은
  대체 대신) 간단히 "뒤집기 예측으로 여성을 남성 취급"했을 때의 TPR 격차 참고치

해석: flip_rate_y1이 현 DEO_lit(0.096)에 비해 작으면, 격차의 대부분은 sex 토큰이
아니라 대리변수(relationship Husband/Wife, marital-status 등)를 타고 있다는 뜻 —
sex 직접 경로만 편집해서는 DEO<0.05 목표에 못 미칠 수 있음을 사전에 알려준다.

출력(--output_dir): metrics.json, flip_predictions.csv (index, pred, pred_flipped)
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

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import format_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",
                   default=str(BIAS_ROOT / "outputs/finetuned_xlm-roberta-large/best"))
    p.add_argument("--test_path", default=str(BIAS_ROOT / "data/test/text/test.csv"))
    p.add_argument("--output_dir",
                   default=str(BIAS_ROOT / "result/llm_base_training/sex_flip_probe"))
    p.add_argument("--flip_mode", choices=["sex", "sex_relationship"], default="sex",
                   help="sex: sex 토큰만 스왑 / sex_relationship: sex + "
                        "relationship(Husband<->Wife) 동시 스왑 (성별 대리 토큰 묶음)")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def build_flipped(df, mode):
    flipped = df.copy()
    flipped["sex"] = df["sex"].map({"Male": "Female", "Female": "Male"})
    if mode == "sex_relationship":
        flipped["relationship"] = df["relationship"].replace(
            {"Husband": "Wife", "Wife": "Husband"})
    return flipped


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size, device):
    preds = []
    for s in tqdm(range(0, len(texts), batch_size), desc="Predict", leave=False):
        enc = tokenizer(texts[s : s + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.test_path)
    flipped = build_flipped(df, args.flip_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    pred_o = predict(model, tokenizer, df.apply(format_prompt, axis=1).tolist(),
                     args.batch_size, device)
    pred_f = predict(model, tokenizer, flipped.apply(format_prompt, axis=1).tolist(),
                     args.batch_size, device)

    y = df["label"].to_numpy()
    male = (df["sex"] == "Male").to_numpy()
    diff = pred_o != pred_f

    # 뒤집었을 때 여성(원본)이 남성 취급을 받으면 TPR이 얼마나 오르나
    tpr_f_orig = float((pred_o[~male & (y == 1)] == 1).mean())
    tpr_f_as_male = float((pred_f[~male & (y == 1)] == 1).mean())
    tpr_m_orig = float((pred_o[male & (y == 1)] == 1).mean())
    tpr_m_as_female = float((pred_f[male & (y == 1)] == 1).mean())

    metrics = {
        "model": args.model_name,
        "flip_mode": args.flip_mode,
        "flip_rate": float(diff.mean()),
        "flip_rate_y1": float(diff[y == 1].mean()),
        "male_to_female_1to0": float(((pred_o == 1) & (pred_f == 0))[male].mean()),
        "female_to_male_0to1": float(((pred_o == 0) & (pred_f == 1))[~male].mean()),
        "tpr_male_orig": tpr_m_orig,
        "tpr_female_orig": tpr_f_orig,
        "deo_lit_orig": abs(tpr_m_orig - tpr_f_orig),
        "tpr_female_if_treated_as_male": tpr_f_as_male,
        "tpr_male_if_treated_as_female": tpr_m_as_female,
        # sex 토큰을 완전히 중립화(직접 경로 제거)해도 남는 격차의 근사:
        # 두 방향 예측의 평균으로 성별 무관 예측을 흉내낸 뒤의 TPR 격차
        "deo_lit_if_sex_token_neutralized": abs(
            float((((pred_o + pred_f) / 2)[male & (y == 1)] >= 0.5).mean())
            - float((((pred_o + pred_f) / 2)[~male & (y == 1)] >= 0.5).mean())
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"index": df.index, "pred": pred_o, "pred_flipped": pred_f}).to_csv(
        out_dir / "flip_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
