"""Exp2-Bias: Exp1 후보 위치에서 실제 causal patch 검증.

각 후보 (layer, position)에서 두 가지를 검증한다:
  (a) 양성: 비혼(unmarried) 문맥에 기혼(married) donor를 patch -> 예측이 승인(1) 쪽으로
      움직이는가 (비혼 페널티 완화 편집이 이 위치에서 실제로 가능한지)
  (b) 음성(분리 검증): cg_below 문맥에 cg_above donor를 patch(같은 marital 위치!) ->
      예측이 거의 안 움직여야 함 (이 위치가 cg rule 판정과는 무관해야 하므로)

Exp1과 마찬가지로 train.csv만 사용(test.csv 무사용).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


import bias_centroid_common as bc  # noqa: E402
import bias_gen_pairs as bp  # noqa: E402

POSITIONS = ["marital-status", "relationship"]


def get_layer_module(model, layer_idx: int):
    if layer_idx == 0:
        return model.roberta.embeddings
    return model.roberta.encoder.layer[layer_idx - 1]


def build_batch_infos(tokenizer, df):
    texts, infos = [], []
    for _, row in df.iterrows():
        text, info = bc.row_positions(tokenizer, row)
        texts.append(text)
        infos.append(info)
    return texts, infos


def build_batch(tokenizer, infos, device):
    ids = [info["input_ids"] for info in infos]
    enc = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def make_patch_hook(idxs_per_sample, donor_per_sample):
    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for i, idxs in enumerate(idxs_per_sample):
            h[i, idxs, :] = donor_per_sample[i].to(h.dtype).to(h.device)
        return (h,) + tuple(output[1:]) if is_tuple else h
    return hook


@torch.no_grad()
def predict_batch(model, batch):
    return model(**batch).logits.argmax(-1).cpu().numpy()


@torch.no_grad()
def predict_patched(model, batch, module, idxs_per_sample, donor_per_sample):
    hook = make_patch_hook(idxs_per_sample, donor_per_sample)
    handle = module.register_forward_hook(hook)
    try:
        out = model(**batch)
    finally:
        handle.remove()
    return out.logits.argmax(-1).cpu().numpy()


def run(args):
    out_dir = bc.RESULT_ROOT / "causal_patch"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    print(f"device={device} n={args.n} layers={args.layers}")

    train = bc.load_train_csv()
    married, unmarried, _ = bp.sample_marital_pairs(train, n=args.n, seed=args.seed)
    cg_above, cg_below, _ = bp.sample_cg_pairs(train, n=args.n, seed=args.seed + 1000)

    tokenizer, model = bc.load_model(device=device)

    # 위치별 hidden state(전체 layer) 미리 추출 -- 후보마다 재추출 안 함
    t0 = time.time()
    h_married, _, _ = bc.extract_multi_position_hidden(model, tokenizer, married, device,
                                                         batch_size=args.batch_size, positions=POSITIONS)
    h_unmarried, texts_u, infos_u = bc.extract_multi_position_hidden(
        model, tokenizer, unmarried, device, batch_size=args.batch_size, positions=POSITIONS)
    h_above, _, _ = bc.extract_multi_position_hidden(model, tokenizer, cg_above, device,
                                                       batch_size=args.batch_size, positions=POSITIONS)
    h_below, texts_b, infos_b = bc.extract_multi_position_hidden(
        model, tokenizer, cg_below, device, batch_size=args.batch_size, positions=POSITIONS)
    print(f"donor/destination hidden state 추출 완료 ({time.time()-t0:.1f}s)")

    # 배치(패딩 통일) -- unmarried, cg_below 각각 한 번씩만 구성해 재사용
    _, infos_u = build_batch_infos(tokenizer, unmarried)
    _, infos_b = build_batch_infos(tokenizer, cg_below)
    batch_u = build_batch(tokenizer, infos_u, device)
    batch_b = build_batch(tokenizer, infos_b, device)

    base_pred_u = predict_batch(model, batch_u)  # 비혼 문맥, patch 전
    base_pred_b = predict_batch(model, batch_b)  # cg_below 문맥, patch 전
    print(f"baseline: unmarried P(pred=1)={base_pred_u.mean():.4f}  "
          f"cg_below P(pred=1)={base_pred_b.mean():.4f}")

    rows = []
    for layer in args.layers:
        module = get_layer_module(model, layer)
        for pos in POSITIONS:
            idxs_u = [info[pos] for info in infos_u]
            idxs_b = [info[pos] for info in infos_b]

            # (a) 양성: unmarried -> married donor
            donor_married = [h_married[pos][i, layer, :] for i in range(len(married))]
            pred_patched_marital = predict_patched(model, batch_u, module, idxs_u, donor_married)
            marital_flip_to_1 = float(((base_pred_u == 0) & (pred_patched_marital == 1)).sum()
                                       / max((base_pred_u == 0).sum(), 1))
            marital_p1_shift = float(pred_patched_marital.mean() - base_pred_u.mean())

            # (b) 음성/분리: cg_below -> cg_above donor (같은 marital 위치에!)
            donor_above = [h_above[pos][i, layer, :] for i in range(len(cg_above))]
            pred_patched_cg = predict_patched(model, batch_b, module, idxs_b, donor_above)
            cg_p1_shift = float(pred_patched_cg.mean() - base_pred_b.mean())
            cg_flip_rate = float((pred_patched_cg != base_pred_b).mean())

            rows.append({
                "layer": layer, "position": pos,
                "marital_p1_shift": marital_p1_shift, "marital_flip_0to1_rate": marital_flip_to_1,
                "cg_p1_shift": cg_p1_shift, "cg_flip_rate": cg_flip_rate,
            })
            print(f"  layer={layer:2d} pos={pos:15s} | marital: P(1) shift={marital_p1_shift:+.4f} "
                  f"flip0->1={marital_flip_to_1:.4f} | cg(분리 검증): P(1) shift={cg_p1_shift:+.4f} "
                  f"flip_rate={cg_flip_rate:.4f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / f"patch_seed{args.seed}.csv", index=False)
    result_df["dissociation_score"] = result_df.marital_p1_shift.abs() - result_df.cg_p1_shift.abs()
    best = result_df.sort_values("dissociation_score", ascending=False)
    print("\n=== 분리 점수(|marital shift| - |cg shift|) 상위 ===")
    print(best[["layer", "position", "marital_p1_shift", "cg_p1_shift", "dissociation_score"]]
          .to_string(index=False))
    best.to_json(out_dir / f"ranked_seed{args.seed}.json", orient="records", indent=2)
    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
