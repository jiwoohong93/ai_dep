"""Exp 2 (EDRM centroid): 후보 위치 인과 검증 (activation patching).

Exp1 히트맵의 상위 5개뿐 아니라 (cls, debt_span) x 레이어 0~24 전체에 대해 flip rate를
측정한다 (사용자 요청으로 확장). 각 (layer, position, direction, patch_type) 조합마다:

  - direction: p2f(통과->탈락, A그룹 100개에 B쪽 표현을 주입) / f2p(탈락->통과, 반대)
  - patch_type: centroid(그룹 평균 벡터 주입) / paired(무작위로 매칭된 상대 그룹 샘플의
    실제 activation 주입) — 두 값의 차이를 함께 기록해 "centroid로 충분한지 vs 실제
    샘플 activation이 필요한지"를 구분한다.

추가로 L22~24 x cls의 표준화 거리 차원별 기여도를 분석해, 상위 10개 차원만 patch했을 때
(sparse patch) 전체 patch와 비슷한 flip rate가 나오는지 확인한다 (rank-1 편집 근거).

산출물 (result/centroid/causal_patch/):
  - rule{r}_flip_full.csv         : layer x position x direction x patch_type flip rate 전체
  - rule{r}_flip_curve_cls.png / rule{r}_flip_curve_debt_span.png
  - rule{r}_dim_contribution.json : L22~24 cls 상위10 차원 기여도 + sparse patch flip rate
  - rule{r}_dissociation.json     : probe 높음(>=0.95) & flip 낮음(<0.3) 위치 목록
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False


import centroid_common as cc  # noqa: E402


# --------------------------------------------------------------------------
# hook 유틸
# --------------------------------------------------------------------------
def get_layer_module(model, layer_idx: int):
    """layer_idx=0 -> embeddings 출력(hidden_states[0]), 1..24 -> encoder.layer[layer_idx-1] 출력."""
    if layer_idx == 0:
        return model.roberta.embeddings
    return model.roberta.encoder.layer[layer_idx - 1]


def make_full_patch_hook(idxs_per_sample, donor_per_sample):
    """destination의 idxs 위치 전체를 donor 벡터로 교체 (broadcast)."""
    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for i, idxs in enumerate(idxs_per_sample):
            h[i, idxs, :] = donor_per_sample[i].to(h.dtype).to(h.device)
        return (h,) + tuple(output[1:]) if is_tuple else h
    return hook


def make_sparse_patch_hook(idxs_per_sample, donor_per_sample, dims):
    """destination의 idxs 위치에서 지정된 dims 차원만 donor 값으로 교체 (나머지 유지)."""
    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for i, idxs in enumerate(idxs_per_sample):
            for t in idxs:
                h[i, t, dims] = donor_per_sample[i][dims].to(h.dtype).to(h.device)
        return (h,) + tuple(output[1:]) if is_tuple else h
    return hook


@torch.no_grad()
def run_with_hook(model, layer_idx, batch, hook_fn):
    module = get_layer_module(model, layer_idx)
    handle = module.register_forward_hook(hook_fn)
    try:
        out = model(**batch)
    finally:
        handle.remove()
    return out.logits.argmax(-1).cpu().numpy()


def build_batch(tokenizer, infos, device):
    ids = [info["input_ids"] for info in infos]
    enc = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def idxs_for_position(info, position):
    return [0] if position == "cls" else info["value_all"]


# --------------------------------------------------------------------------
# 메인 스윕
# --------------------------------------------------------------------------
def run(args):
    cc.ensure_dirs()
    out_dir = cc.RESULT_ROOT / "causal_patch"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, rule_column, theta, *_ = cc.rule_spec(args.rule)
    device = f"cuda:{args.gpu_id}"
    print(f"rule={args.rule} ({rule_column}, theta={theta})  device={device}")

    df = cc.load_test_csv()
    # Exp1과 동일한 seed/n_per_group으로 재현 -> 캐시된 500개 hidden state와 정확히 일치
    df_a, df_b = cc.decisive_split(df, args.rule, n_per_group=500, seed=0)

    tokenizer, model = cc.load_model(args.model_name, device=device)

    cache_a = cc.CACHE_DIR / f"locate_scan_rule{args.rule}_A.pt"
    cache_b = cc.CACHE_DIR / f"locate_scan_rule{args.rule}_B.pt"
    cls_a, span_a = cc.get_cached_hidden(cache_a, model, tokenizer, df_a, rule_column, device)
    cls_b, span_b = cc.get_cached_hidden(cache_b, model, tokenizer, df_b, rule_column, device)
    full_hidden = {"cls": (cls_a, cls_b), "debt_span": (span_a, span_b)}

    rng = np.random.default_rng(args.seed)
    n_patch = args.n_patch
    df_a_sub = df_a.head(n_patch).reset_index(drop=True)
    df_b_sub = df_b.head(n_patch).reset_index(drop=True)
    print(f"patch 대상: A {len(df_a_sub)}건, B {len(df_b_sub)}건 (전체 A={len(df_a)}, B={len(df_b)})")

    print("토큰 span 계산...")
    infos_a = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in df_a_sub.iterrows()]
    infos_b = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in df_b_sub.iterrows()]
    batch_a = build_batch(tokenizer, infos_a, device)
    batch_b = build_batch(tokenizer, infos_b, device)

    with torch.no_grad():
        clean_pred_a = model(**batch_a).logits.argmax(-1).cpu().numpy()
        clean_pred_b = model(**batch_b).logits.argmax(-1).cpu().numpy()
    valid_a = clean_pred_a == 1  # A는 '통과(1)'로 정답 예측한 것만 p2f 대상
    valid_b = clean_pred_b == 0  # B는 '탈락(0)'으로 정답 예측한 것만 f2p 대상
    print(f"clean 정답 예측: A(valid)={valid_a.sum()}/{len(valid_a)}  B(valid)={valid_b.sum()}/{len(valid_b)}")

    # paired donor 인덱스 (전체 500 중에서 무작위 매칭, subsample 크기만큼)
    paired_donor_idx_for_a = rng.integers(0, len(df_b), size=n_patch)  # A 각각에 매칭될 B(전체) 인덱스
    paired_donor_idx_for_b = rng.integers(0, len(df_a), size=n_patch)

    layers = list(range(cc.N_HIDDEN_STATES))
    rows = []
    t0 = time.time()
    for position in ("cls", "debt_span"):
        ha_full, hb_full = full_hidden[position]  # (500, 25, H) each
        idxs_a = [idxs_for_position(info, position) for info in infos_a]
        idxs_b = [idxs_for_position(info, position) for info in infos_b]
        for layer in layers:
            centroid_b = hb_full[:, layer, :].mean(0)  # (H,)
            centroid_a = ha_full[:, layer, :].mean(0)

            # ---- centroid patch ----
            donor_b_for_a = [centroid_b] * n_patch
            pred = run_with_hook(model, layer, batch_a, make_full_patch_hook(idxs_a, donor_b_for_a))
            flip_p2f_centroid = float((pred[valid_a] == 0).mean()) if valid_a.sum() else float("nan")

            donor_a_for_b = [centroid_a] * n_patch
            pred = run_with_hook(model, layer, batch_b, make_full_patch_hook(idxs_b, donor_a_for_b))
            flip_f2p_centroid = float((pred[valid_b] == 1).mean()) if valid_b.sum() else float("nan")

            # ---- paired patch (무작위 매칭된 상대 그룹 실제 activation) ----
            donor_b_for_a_paired = [hb_full[j, layer, :] for j in paired_donor_idx_for_a]
            pred = run_with_hook(model, layer, batch_a, make_full_patch_hook(idxs_a, donor_b_for_a_paired))
            flip_p2f_paired = float((pred[valid_a] == 0).mean()) if valid_a.sum() else float("nan")

            donor_a_for_b_paired = [ha_full[j, layer, :] for j in paired_donor_idx_for_b]
            pred = run_with_hook(model, layer, batch_b, make_full_patch_hook(idxs_b, donor_a_for_b_paired))
            flip_f2p_paired = float((pred[valid_b] == 1).mean()) if valid_b.sum() else float("nan")

            rows.append({
                "layer": layer, "position": position,
                "flip_p2f_centroid": flip_p2f_centroid, "flip_f2p_centroid": flip_f2p_centroid,
                "flip_p2f_paired": flip_p2f_paired, "flip_f2p_paired": flip_f2p_paired,
                "diff_p2f_paired_minus_centroid": flip_p2f_paired - flip_p2f_centroid,
                "diff_f2p_paired_minus_centroid": flip_f2p_paired - flip_f2p_centroid,
                "n_valid_a": int(valid_a.sum()), "n_valid_b": int(valid_b.sum()),
            })
            print(f"  L{layer:2d} {position:9s} | centroid p2f={flip_p2f_centroid:.3f} f2p={flip_f2p_centroid:.3f} "
                  f"| paired p2f={flip_p2f_paired:.3f} f2p={flip_f2p_paired:.3f}")

    print(f"스윕 완료 ({time.time()-t0:.1f}s)")
    flip_df = pd.DataFrame(rows)
    flip_df.to_csv(out_dir / f"rule{args.rule}_flip_full.csv", index=False)

    # ---------------- flip curve 플롯 ----------------
    for position in ("cls", "debt_span"):
        sub = flip_df[flip_df["position"] == position].sort_values("layer")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["layer"], sub["flip_p2f_centroid"], marker="o", label="centroid patch, 통과→탈락(p2f)", color="#c0392b")
        ax.plot(sub["layer"], sub["flip_f2p_centroid"], marker="o", label="centroid patch, 탈락→통과(f2p)", color="#2980b9")
        ax.plot(sub["layer"], sub["flip_p2f_paired"], marker="s", linestyle="--", label="paired patch, p2f", color="#e67e22")
        ax.plot(sub["layer"], sub["flip_f2p_paired"], marker="s", linestyle="--", label="paired patch, f2p", color="#5dade2")
        ax.set_xlabel("layer (0=embedding, 24=final)")
        ax.set_ylabel("flip rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"rule{args.rule} ({rule_column}) — {position} activation patching flip rate")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"rule{args.rule}_flip_curve_{position}.png", dpi=150)
        plt.close(fig)

    # ---------------- 차원 기여도 분석 (L22~24, cls) ----------------
    dim_report = {}
    cls_a_full, cls_b_full = full_hidden["cls"]
    for layer in args.dim_layers:
        xa = cls_a_full[:, layer, :].numpy()
        xb = cls_b_full[:, layer, :].numpy()
        mu_a, mu_b = xa.mean(0), xb.mean(0)
        var_a, var_b = xa.var(0), xb.var(0)
        pooled_std = np.clip(np.sqrt((var_a + var_b) / 2.0), 1e-6, None)
        z = (mu_a - mu_b) / pooled_std
        order = np.argsort(-np.abs(z))
        top10 = order[:10]
        contribution_ratio = float((z[top10] ** 2).sum() / (z ** 2).sum())

        centroid_b = cls_b_full[:, layer, :].mean(0)
        centroid_a = cls_a_full[:, layer, :].mean(0)
        idxs_a_cls = [[0]] * n_patch
        idxs_b_cls = [[0]] * n_patch

        pred = run_with_hook(model, layer, batch_a, make_full_patch_hook(idxs_a_cls, [centroid_b] * n_patch))
        flip_full_p2f = float((pred[valid_a] == 0).mean()) if valid_a.sum() else float("nan")
        pred = run_with_hook(model, layer, batch_b, make_full_patch_hook(idxs_b_cls, [centroid_a] * n_patch))
        flip_full_f2p = float((pred[valid_b] == 1).mean()) if valid_b.sum() else float("nan")

        pred = run_with_hook(model, layer, batch_a,
                              make_sparse_patch_hook(idxs_a_cls, [centroid_b] * n_patch, top10))
        flip_sparse_p2f = float((pred[valid_a] == 0).mean()) if valid_a.sum() else float("nan")
        pred = run_with_hook(model, layer, batch_b,
                              make_sparse_patch_hook(idxs_b_cls, [centroid_a] * n_patch, top10))
        flip_sparse_f2p = float((pred[valid_b] == 1).mean()) if valid_b.sum() else float("nan")

        dim_report[f"layer_{layer}_cls"] = {
            "top10_dims": top10.tolist(),
            "top10_abs_z": np.abs(z[top10]).tolist(),
            "contribution_ratio_top10": contribution_ratio,
            "flip_full_p2f": flip_full_p2f, "flip_full_f2p": flip_full_f2p,
            "flip_sparse_top10_p2f": flip_sparse_p2f, "flip_sparse_top10_f2p": flip_sparse_f2p,
        }
        print(f"[dim contribution] layer={layer} cls: top10 기여율={contribution_ratio:.3f} "
              f"| full flip p2f={flip_full_p2f:.3f} f2p={flip_full_f2p:.3f} "
              f"| sparse(top10) flip p2f={flip_sparse_p2f:.3f} f2p={flip_sparse_f2p:.3f}")

    with open(out_dir / f"rule{args.rule}_dim_contribution.json", "w", encoding="utf-8") as f:
        json.dump(dim_report, f, ensure_ascii=False, indent=2)

    # ---------------- 해리(dissociation) 목록: probe 높음 & flip 낮음 ----------------
    scan_path = cc.RESULT_ROOT / "locate_scan" / f"rule{args.rule}_scan_full.csv"
    dissociation = []
    if scan_path.exists():
        scan_df = pd.read_csv(scan_path)
        merged = scan_df.merge(flip_df, on=["layer", "position"], how="left")
        mask = (
            (merged["probe_acc"] >= args.probe_threshold)
            & (merged["flip_p2f_centroid"] < args.flip_threshold)
            & (merged["flip_f2p_centroid"] < args.flip_threshold)
        )
        dissociation = merged[mask][[
            "layer", "position", "probe_acc", "standardized_distance",
            "flip_p2f_centroid", "flip_f2p_centroid",
        ]].to_dict(orient="records")
        merged.to_csv(out_dir / f"rule{args.rule}_scan_flip_merged.csv", index=False)

    with open(out_dir / f"rule{args.rule}_dissociation.json", "w", encoding="utf-8") as f:
        json.dump({
            "probe_threshold": args.probe_threshold, "flip_threshold": args.flip_threshold,
            "n_dissociation": len(dissociation), "positions": dissociation,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n해리(probe>={args.probe_threshold} & flip<{args.flip_threshold}) 위치: {len(dissociation)}건")
    for d in dissociation[:15]:
        print(f"  L{d['layer']:2d} {d['position']:9s} probe={d['probe_acc']:.3f} "
              f"flip_p2f={d['flip_p2f_centroid']:.3f} flip_f2p={d['flip_f2p_centroid']:.3f}")

    print(f"\n저장 완료: {out_dir}")
    return flip_df, dim_report, dissociation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--n_patch", type=int, default=100)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim_layers", type=int, nargs="+", default=[22, 23, 24])
    parser.add_argument("--probe_threshold", type=float, default=0.95)
    parser.add_argument("--flip_threshold", type=float, default=0.3)
    parser.add_argument("--model_name", default=cc.LOCAL_MODEL_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
