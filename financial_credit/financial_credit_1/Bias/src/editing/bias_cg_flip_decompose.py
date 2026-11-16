"""band(253명) 중 flip 안 된 ~41%의 특성 분석 -- "다른 고소득 신호가 강해서 cg 신호 하나
되돌려도 안 뒤집힌다"는 신호-중복 가설을 직접 검증.

flip 그룹 vs 비flip 그룹으로 education-num/occupation/hours-per-week/age 등을 비교한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


import bias_centroid_common as bc  # noqa: E402
from bias_centroid_hooks import get_layer_module, build_batch_infos, build_batch  # noqa: E402
from bias_centroid_edit_cg import GatedShiftCgHook, OLD_THETA, NEW_THETA  # noqa: E402

OUT_DIR = bc.RESULT_ROOT / "cg_flip_decompose"


@torch.no_grad()
def predict_full(model, tokenizer, df, device, module=None, hook_factory=None, batch_size=64):
    preds = []
    rows = df.reset_index(drop=True)
    for i in range(0, len(rows), batch_size):
        sub = rows.iloc[i:i + batch_size]
        texts, infos = build_batch_infos(tokenizer, sub)
        batch = build_batch(tokenizer, infos, device)
        handle = None
        if hook_factory is not None:
            idxs = [info["capital-gain"] for info in infos]
            handle = module.register_forward_hook(hook_factory(idxs))
        try:
            out = model(**batch)
        finally:
            if handle is not None:
                handle.remove()
        preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def run(cg_config, alpha, gpu_id, batch_size=64):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu_id}"
    cfg = torch.load(cg_config)
    layer = cfg["layer"]

    df = bc.load_test_csv()
    cg = df["capital-gain"].to_numpy()
    band_mask = (cg >= OLD_THETA) & (cg < NEW_THETA)
    band_df = df[band_mask].reset_index(drop=True)
    print(f"band n={len(band_df)}")

    tokenizer, model = bc.load_model(device=device)
    module = get_layer_module(model, layer)

    pred_base = predict_full(model, tokenizer, band_df, device, batch_size=batch_size)

    def hook_factory(idxs):
        return GatedShiftCgHook(idxs, cfg["w_gate"], cfg["b_gate"], cfg["delta_v"], alpha)
    pred_edit = predict_full(model, tokenizer, band_df, device, module=module, hook_factory=hook_factory,
                              batch_size=batch_size)

    flipped = (pred_base == 1) & (pred_edit == 0)
    unflipped = (pred_base == 1) & (pred_edit == 1)
    print(f"flip={flipped.sum()} unflip={unflipped.sum()} (base pred=0인 {int((pred_base==0).sum())}명은 제외)")

    numeric_cols = ["age", "education-num", "hours-per-week", "fnlwgt", "capital-loss"]
    cat_cols = ["occupation", "education", "workclass", "relationship", "sex"]

    summary = {"n_flipped": int(flipped.sum()), "n_unflipped": int(unflipped.sum())}
    print("\n=== 수치형 특성 평균 비교 (flip vs unflip) ===")
    numeric_rows = []
    for col in numeric_cols:
        m_f = float(band_df.loc[flipped, col].mean())
        m_u = float(band_df.loc[unflipped, col].mean())
        numeric_rows.append({"feature": col, "mean_flipped": m_f, "mean_unflipped": m_u, "diff(unflip-flip)": m_u - m_f})
        print(f"  {col:16s} flip={m_f:10.2f}  unflip={m_u:10.2f}  diff(unflip-flip)={m_u-m_f:+8.2f}")
    summary["numeric"] = numeric_rows

    print("\n=== 범주형 특성 상위값 비교 (flip vs unflip, 비율) ===")
    cat_summary = {}
    for col in cat_cols:
        vf = band_df.loc[flipped, col].value_counts(normalize=True).head(5)
        vu = band_df.loc[unflipped, col].value_counts(normalize=True).head(5)
        print(f"  [{col}] flip 상위: {vf.to_dict()}")
        print(f"  [{col}] unflip 상위: {vu.to_dict()}")
        cat_summary[col] = {"flipped": vf.to_dict(), "unflipped": vu.to_dict()}
    summary["categorical"] = cat_summary

    # 학사 이상 학력 비율(고소득 신호 프록시)
    high_edu = {"Bachelors", "Masters", "Doctorate", "Prof-school"}
    prof_occ = {"Prof-specialty", "Exec-managerial"}
    hi_edu_f = float(band_df.loc[flipped, "education"].isin(high_edu).mean())
    hi_edu_u = float(band_df.loc[unflipped, "education"].isin(high_edu).mean())
    prof_f = float(band_df.loc[flipped, "occupation"].isin(prof_occ).mean())
    prof_u = float(band_df.loc[unflipped, "occupation"].isin(prof_occ).mean())
    print(f"\n고학력(학사+) 비율: flip={hi_edu_f:.3f} unflip={hi_edu_u:.3f}")
    print(f"전문/관리직 비율: flip={prof_f:.3f} unflip={prof_u:.3f}")
    summary["high_edu_rate"] = {"flipped": hi_edu_f, "unflipped": hi_edu_u}
    summary["prof_occ_rate"] = {"flipped": prof_f, "unflipped": prof_u}

    with open(OUT_DIR / f"decompose_alpha{alpha}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    band_df["flipped"] = flipped
    band_df["unflipped"] = unflipped
    band_df.to_csv(OUT_DIR / f"band_rows_alpha{alpha}.csv", index=False)
    print(f"\n저장 완료: {OUT_DIR}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cg_config",
                    default=str(bc.RESULT_ROOT / "edit_cg/layer15_seed0/shift_config.pt"))
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--gpu_id", type=int, default=0)
    args = p.parse_args()
    run(args.cg_config, args.alpha, args.gpu_id)
