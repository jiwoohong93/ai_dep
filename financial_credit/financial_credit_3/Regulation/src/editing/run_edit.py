"""편집 실행 → 즉시 평가 → 델타·지표 저장 (단일 CLI).

사용 예:
    python src/editing/run_edit.py --case 1 --method rome
층은 기본적으로 locate 결과(result/locate/rule{case}/n10000/meta.json)의 p2f 피크.
MEMIT/AlphaEdit은 [피크−2, min(피크+2, 22)] 구간에 잔차를 균등 분산.

평가: test_case{i}.csv 전체 + 플립/retain 분해(pre-edit 예측과 행 단위 대조)
      + θ′ 경계 generalization (합성 2k: 플립존→0, 안전존→1)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.editing import editors, kv
from src.models.test import format_prompt


def parse_args():
    p = argparse.ArgumentParser(description="Apply a baseline edit and evaluate.")
    p.add_argument("--case", type=int, required=True, choices=range(1, 10))
    p.add_argument("--method", required=True, choices=["rome", "memit", "emmet", "alphaedit"])
    p.add_argument("--model_name", default="yongukpark/xlm-roberta-large-rule")
    p.add_argument("--layer", type=int, default=None, help="기본: locate p2f 피크")
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--context_mode", default="fixed_partner",
                   choices=["fixed_partner", "random_pass"])
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--locate_dir", default="result/locate")
    p.add_argument("--output_root", default="result/edit")
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--skip_eval", action="store_true", help="편집·검증만 (파이프라인 디버그용)")
    return p.parse_args()


def peak_layer(locate_dir, case):
    meta = json.load(open(Path(locate_dir) / f"rule{case}" / "n10000" / "meta.json"))
    return meta["peak_layer_p2f"]


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size, device):
    preds = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Predict", leave=False):
        enc = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt", padding=True, truncation=True, max_length=256,
        ).to(device)
        preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def evaluate(model, tokenizer, case, batch_size, device, rng):
    """test_case 전체 + 플립/retain 분해 + 경계 generalization."""
    base = pd.read_csv("data/test/text/test.csv", low_memory=False).dropna()
    edited = pd.read_csv(
        f"data/test/text/edited_rule_25pct/test_case{case}.csv", low_memory=False
    ).dropna()
    assert (base.index == edited.index).all()
    texts = base.apply(format_prompt, axis=1).tolist()
    preds = predict(model, tokenizer, texts, batch_size, device)

    label_new = edited["label"].astype(int).to_numpy()
    flip = (base["label"].astype(int).to_numpy() != label_new)

    pre = pd.read_csv(
        f"result/evaluation/edited_rule_25pct/case{case}/predictions.csv"
    ).set_index("index").loc[base.index, "pred"].to_numpy()
    pre_overall = float((pre == label_new).mean())
    pre_retain = float((pre[~flip] == label_new[~flip]).mean())

    # 경계 generalization (합성): 플립존 1k → 0, 안전존 1k → 1
    gen_metrics = {}
    for zone, target in [("flip", 0), ("safe", 1)]:
        rows = kv.build_rows(case, kv.sample_zone_values(case, zone, 1000, rng), rng)
        gtexts = rows.apply(format_prompt, axis=1).tolist()
        gpreds = predict(model, tokenizer, gtexts, batch_size, device)
        gen_metrics[f"gen_{zone}_acc"] = float((gpreds == target).mean())

    overall = float((preds == label_new).mean())
    locality = float((preds[~flip] == label_new[~flip]).mean())
    return {
        "overall_acc": overall,
        "efficacy_flip_acc": float((preds[flip] == label_new[flip]).mean()),
        "locality_retain_acc": locality,
        "locality_pred_unchanged": float((preds[~flip] == pre[~flip]).mean()),
        # 편집 전(pre-edit) 대비: overall은 개선폭(+), retain은 하락률(−)
        "preedit_overall_acc": pre_overall,
        "overall_acc_change": overall - pre_overall,
        "preedit_retain_acc": pre_retain,
        "retain_acc_drop": pre_retain - locality,
        "n_flip": int(flip.sum()), "n_retain": int((~flip).sum()),
        **gen_metrics,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed + args.case)
    layer = args.layer if args.layer is not None else peak_layer(args.locate_dir, args.case)
    spec = kv.CASE_SPECS[args.case]
    t0 = time.time()
    print(f"case {args.case} ({spec['column']} {spec['theta']:g}→{spec['theta_new']:g}) | "
          f"method={args.method} layer=L{layer} R={args.R}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    # 1) 편집 재료 (전부 합성)
    inputs = kv.build_edit_inputs(
        args.case, tokenizer, rng, n_requests=args.R, context_mode=args.context_mode
    )
    req_texts, req_pos, _ = inputs["requests"]
    C = kv.compute_covariance(model, tokenizer, inputs["general"], layer)
    v_star, _ = kv.optimize_v_star(model, tokenizer, req_texts, req_pos, layer, target_label=0)
    K, _ = kv.collect_keys_values(model, tokenizer, req_texts, req_pos, layer)
    K = K.T.double().to(device)          # [4096, R]
    V_star = v_star.T.double().to(device)  # [1024, R]
    C = C.double().to(device)

    fc2 = model.roberta.encoder.layer[layer].output.dense
    W, bias = fc2.weight.data.float(), fc2.bias.data.float()

    # 2) 방법별 update
    deltas = {}
    if args.method == "rome":
        new_W = editors.rome_sequential(W, bias, K.float(), V_star.float(), C.float())
        deltas[layer] = editors.apply_edit(model, layer, new_W.to(W.dtype))
    elif args.method == "emmet":
        new_W = editors.emmet_batch(W, bias, K.float(), V_star.float(), C.float())
        deltas[layer] = editors.apply_edit(model, layer, new_W.to(W.dtype))
    elif args.method in ("memit", "alphaedit"):
        window = list(range(max(0, layer - 2), min(22, layer + 2) + 1))
        n = len(window)
        # 잔차(v* − 현재 fc2out at 피크층)를 창의 각 층에 균등 분산 (하위층부터, 층별 키 재수집)
        for i, l in enumerate(window):
            K_l, V_l = kv.collect_keys_values(model, tokenizer, req_texts, req_pos, l)
            _, V_peak = kv.collect_keys_values(model, tokenizer, req_texts, req_pos, layer)
            res_total = v_star - V_peak                     # [R, 1024] 남은 잔차 (피크층 기준)
            target_l = (V_l + res_total / (n - i)).T.float().to(device)  # 이 층에서 흡수할 몫
            C_l = (
                kv.compute_covariance(model, tokenizer, inputs["general"], l)
                if l != layer else C
            ).float().to(device)
            fc2_l = model.roberta.encoder.layer[l].output.dense
            W_l, b_l = fc2_l.weight.data.float(), fc2_l.bias.data.float()
            K_l = K_l.T.float().to(device)
            if args.method == "memit":
                new_W = editors.memit_batch(W_l, b_l, K_l, target_l, C_l, lam=args.lam)
            else:
                K0_ll, _ = kv.collect_keys_values(model, tokenizer, *inputs["preserve"], l)
                new_W = editors.alphaedit_batch(
                    W_l, b_l, K_l, target_l, C_l, K0_ll.T.float().to(device), lam=args.lam
                )
            deltas[l] = editors.apply_edit(model, l, new_W.to(fc2_l.weight.dtype))

    # 3) 편집 자체 검증: 요청 문장들이 실제로 0으로 뒤집혔나 (합성, 평가셋 아님)
    req_preds = predict(model, tokenizer, req_texts, args.eval_batch_size, device)
    edit_success = float((req_preds == 0).mean())
    print(f"편집 요청 충족률(합성 요청 {args.R}개가 0으로): {edit_success*100:.1f}%")

    suffix = "" if args.context_mode == "fixed_partner" else f"_{args.context_mode}"
    out_dir = Path(args.output_root) / args.method / f"case{args.case}_L{layer}_R{args.R}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({l: d.half() for l, d in deltas.items()}, out_dir / "w2_delta.pt")

    metrics = {
        "case": args.case, "method": args.method, "layer": layer, "R": args.R,
        "context_mode": args.context_mode,
        "lam": args.lam, "seed": args.seed, "edit_request_success": edit_success,
        "delta_norm": {l: float(d.norm()) for l, d in deltas.items()},
    }
    # 4) 평가
    if not args.skip_eval:
        model.to(torch.bfloat16)
        metrics.update(evaluate(model, tokenizer, args.case, args.eval_batch_size, device, rng))
        print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, float)},
                         ensure_ascii=False, indent=2))
    metrics["runtime_sec"] = round(time.time() - t0, 1)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
