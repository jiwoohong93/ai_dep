"""Locate 재실험 — 100% 합성 데이터 (test.csv 미사용).

배경: 기존 v1 locate(probe 스캔 + activation patching, centroid_causal_patch.py)는
A/B 그룹 500/500을 test.csv에서 뽑았다(decisive_split). 이는 v2 원칙
("편집 재료 제작에 실데이터 금지, test.csv는 최종 채점 전용")에 어긋나므로,
locate 전체(probe + patching)를 합성 최소쌍만으로 다시 실행해 L21 선정을 재검증한다.

데이터: data/rules/edited_rule_25pct/locate/rule{r}.csv
  - generate_locate_pairs.py가 seed 고정으로 생성한 rule당 10,000 합성 쌍 (git 추적)
  - A그룹(통과) = 쌍의 base 행, B그룹(탈락) = 같은 문맥에서 rule 값만 value_fail로
    바꾼 행 → v1의 "decisive 실데이터"보다 문맥 통제가 더 엄격 (문맥 완전 동일)
  - manifest에 데이터 파일 md5 + subsample seed + git commit 기록 (버전 추적)

절차:
  1) probe 스캔: (layer 0..24) x (cls, debt_span) = 50개 지점 각각에서
     로지스틱 회귀 70/30 held-out 정확도 + 표준화 거리
  2) activation patching: 지점별 x 방향(p2f: A에 B표현 주입 / f2p: 반대)
     x 주입 방식(centroid: 그룹 평균 / paired: 같은 문맥의 반대편 쌍 벡터) flip rate
     ※ v1은 paired 상대를 무작위 매칭했지만, 최소쌍에서는 자기 자신의 반대편
       (같은 문맥의 counterfactual)이 자연스러운 상대라 그걸 쓴다.
  3) 선별: probe_acc >= PROBE_MIN 이면서 양방향 centroid flip >= FLIP_MIN 인 지점을
     후보로, probe만 높고 flip 낮은 지점(해리)은 배제 목록으로 저장.

실행 (Regulation/centroid/src/editing/ 에서):
  CUDA_VISIBLE_DEVICES=3 python centroid_locate_synth.py --rule 1
"""
import argparse
import hashlib
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["axes.unicode_minus"] = False
try:
    matplotlib.rcParams["font.family"] = "NanumGothic"
except Exception:
    pass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import centroid_common as cc  # noqa: E402
import centroid_v2_common as v2  # noqa: E402
from centroid_causal_patch import (  # noqa: E402
    get_layer_module, make_full_patch_hook, run_with_hook, build_batch, idxs_for_position,
)

PROBE_MIN = 0.95   # 후보 조건 1: 정보가 담겨 있음
FLIP_MIN = 0.70    # 후보 조건 2: 양방향 centroid flip
FLIP_DISS = 0.30   # 해리 조건: probe 높은데 양방향 flip이 이보다 낮음

# locate 데이터의 문맥 레시피를 edit(gate) 버전과 1:1로 맞춘다 (locate v3a ↔ edit v3a):
#   pairs: 기존 합성 최소쌍 (partner-cycle 좁은 문맥) — v2 delta_v의 출처, 참고 기준선
#   v3a  : gate_v3a와 동일 재료 — result/centroid/gate_v3a/data/(gaussian+interpolated) 문맥
#   v3b  : gate_v3b와 동일 재료 — 다른 8개 rule 안전존 전체 균일 + 정보성 4개 넓은 gaussian
#   v3b2 : gate_v3b2와 동일 재료 — v3b + 25% 행은 다른 rule 1~2개를 일부러 탈락값으로
CONTEXT_MODES = ("pairs", "v3a", "v3b", "v3b2")


def build_context_pairs(rule_num: int, n_pairs: int, seed: int, mode: str):
    """해당 버전의 gate 학습 데이터와 같은 레시피의 문맥 위에, rule 값만 통과/탈락 쌍으로
    넣은 A/B 프레임 생성. 반환: (df_a 통과측, df_b 같은 문맥의 탈락측, noisy_mask)."""
    from centroid_v3b_pipeline import SAFE_RANGES, INFO_WIDE_MULT
    from centroid_v3b2_pipeline import FAIL_RANGES, FRAC_NOISY
    from src.editing.generate_locate_pairs import FEATURE_COLUMNS, AMOUNT_COLUMNS, sample_side

    _, target_column, theta, sigma, pass_side, delta, is_amount = cc.rule_spec(rule_num)
    rng = np.random.default_rng(seed)
    noisy_mask = np.zeros(n_pairs, dtype=bool)

    if mode == "v3a":
        # gate_v3a가 쓴 바로 그 데이터(git 추적)에서 문맥을 샘플링 — 레시피가 아니라 파일 자체를 공유.
        # 단, 다른 rule이 탈락인 문맥에서는 부채비율이 판정을 못 바꾸므로(locate 전제 붕괴)
        # gate_v3a의 band/safe 라벨과 동일하게 decisive(타 rule 전부 통과) 문맥만 사용.
        v3a_dir = cc.CENTROID_ROOT / "result" / "centroid" / "gate_v3a" / "data"
        pool = pd.concat([pd.read_csv(v3a_dir / "gaussian.csv"),
                          pd.read_csv(v3a_dir / "interpolated.csv")], ignore_index=True)
        pool = pool[cc.other_rules_pass_mask(pool, target_column)].reset_index(drop=True)
        if len(pool) < n_pairs:
            raise RuntimeError(f"v3a decisive 문맥 부족: {len(pool)} < {n_pairs}")
        sel = rng.choice(len(pool), size=n_pairs, replace=False)
        ctx = pool.iloc[sel].reset_index(drop=True)[
            [c for c in FEATURE_COLUMNS if c != target_column]].copy()
    else:  # v3b / v3b2: v3b 생성기와 동일 규칙으로 문맥 생성
        data = {}
        for col in FEATURE_COLUMNS:
            if col == target_column:
                continue
            if col in SAFE_RANGES:
                lo, hi = SAFE_RANGES[col]
                data[col] = rng.uniform(lo, hi, n_pairs)
            else:
                defaults = {"총자산순이익률ROA(%)": (0.3, 0.3), "영업흑자": (1.0, 500_000.0),
                            "이익잉여금양수": (1.0, 500_000.0), "예대율(%)": (100.0, 15.0)}
                mu, sd = defaults[col]
                vals = rng.normal(mu, sd * INFO_WIDE_MULT, n_pairs)
                if col not in ("영업흑자", "이익잉여금양수"):
                    vals = np.clip(vals, 0.0, None)
                data[col] = vals
        ctx = pd.DataFrame(data)

        if mode == "v3b2":
            n_noisy = int(round(n_pairs * FRAC_NOISY))
            noisy_idx = rng.choice(n_pairs, size=n_noisy, replace=False)
            other_cols = list(SAFE_RANGES.keys())
            for row_i in noisy_idx:
                for col in rng.choice(other_cols, size=int(rng.integers(1, 3)), replace=False):
                    lo, hi = FAIL_RANGES[col]
                    ctx.loc[row_i, col] = rng.uniform(lo, hi)
            noisy_mask[noisy_idx] = True

    fail_side = "above" if pass_side == "below" else "below"
    v_pass = sample_side(rng, theta, sigma, pass_side, delta, is_amount, n_pairs)
    v_fail = sample_side(rng, theta, sigma, fail_side, delta, is_amount, n_pairs)
    df_a = ctx.copy()
    df_a[target_column] = v_pass
    df_b = ctx.copy()
    df_b[target_column] = v_fail

    percent_cols = [c for c in FEATURE_COLUMNS if c not in AMOUNT_COLUMNS]
    for df in (df_a, df_b):
        for col in AMOUNT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col]).round().astype("int64")
        for col in percent_cols:
            # v3a 문맥의 예대율(%)엔 "산정불가" 문자열이 섞여 있을 수 있음(학습 분포와 동일) — 숫자만 반올림
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
    return df_a[FEATURE_COLUMNS], df_b[FEATURE_COLUMNS], noisy_mask


def probe_heldout_acc(x_a: np.ndarray, x_b: np.ndarray, seed: int):
    """지점 1곳의 probe: A/B 벡터로 로지스틱 회귀 학습, 70/30 held-out 정확도."""
    X = np.concatenate([x_a, x_b], axis=0)
    y = np.concatenate([np.ones(len(x_a)), np.zeros(len(x_b))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(scaler.transform(X_tr), y_tr)
    acc = float(clf.score(scaler.transform(X_te), y_te))
    # 표준화 거리 (두 centroid가 pooled std 기준 얼마나 떨어졌나)
    mu_a, mu_b = x_a.mean(0), x_b.mean(0)
    pooled_std = np.clip(np.sqrt((x_a.var(0) + x_b.var(0)) / 2.0), 1e-6, None)
    std_dist = float(np.linalg.norm((mu_a - mu_b) / pooled_std) / np.sqrt(x_a.shape[1]))
    return acc, std_dist


def run(args):
    start_time = datetime.now()
    run_dir = cc.CENTROID_ROOT / "result" / "centroid_matched" / f"locate_synth_{args.context_mode}"
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{run_dir} 이미 존재. --overwrite 로 명시적으로 덮어쓰세요.")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    _, rule_column, theta, *_ = cc.rule_spec(args.rule)
    rng = np.random.default_rng(args.seed)

    # ---------------- locate용 데이터 준비 (모드별 — edit 버전과 레시피 일치) ----------------
    if args.context_mode == "pairs":
        pairs_path = v2.LOCATE_PAIRS_DIR / f"rule{args.rule}.csv"
        data_md5 = hashlib.md5(pairs_path.read_bytes()).hexdigest()
        data_desc = str(pairs_path.relative_to(cc.REGULATION_ROOT))
        base_full, source_full, _ = v2.load_locate_pairs(args.rule)  # 문맥 동일, rule 값만 다름
        sel = rng.choice(len(base_full), size=min(args.n_pairs, len(base_full)), replace=False)
        df_a = base_full.iloc[sel].reset_index(drop=True)    # A = 통과측
        df_b = source_full.iloc[sel].reset_index(drop=True)  # B = 같은 문맥의 탈락측
        noisy_mask = np.zeros(len(df_a), dtype=bool)
        print(f"합성 쌍 {len(base_full)}개 중 {len(df_a)}쌍 subsample (seed={args.seed})")
    else:
        df_a, df_b, noisy_mask = build_context_pairs(args.rule, args.n_pairs, args.seed,
                                                      args.context_mode)
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        df_a.assign(_noisy=noisy_mask).to_csv(data_dir / "pairs_A.csv", index=False)
        df_b.assign(_noisy=noisy_mask).to_csv(data_dir / "pairs_B.csv", index=False)
        data_md5 = hashlib.md5((data_dir / "pairs_A.csv").read_bytes()
                               + (data_dir / "pairs_B.csv").read_bytes()).hexdigest()
        data_desc = f"generated in-run: {args.context_mode} recipe (seed={args.seed})"
        print(f"{args.context_mode} 문맥 쌍 {len(df_a)}개 생성 (noisy={int(noisy_mask.sum())})")
    print(f"rule={args.rule} ({rule_column}, theta={theta}) | mode={args.context_mode} md5={data_md5}")

    tokenizer, model = cc.load_model(args.model_name, device=device)

    # ---------------- hidden state 추출 (50개 지점 전부, 캐시) ----------------
    t0 = time.time()
    cls_a, span_a = cc.get_cached_hidden(run_dir / "cache_A.pt", model, tokenizer, df_a,
                                          rule_column, device, batch_size=args.batch_size)
    cls_b, span_b = cc.get_cached_hidden(run_dir / "cache_B.pt", model, tokenizer, df_b,
                                          rule_column, device, batch_size=args.batch_size)
    full_hidden = {"cls": (cls_a, cls_b), "debt_span": (span_a, span_b)}
    print(f"hidden 추출 완료 ({time.time()-t0:.1f}s)  A={cls_a.shape} B={cls_b.shape}")

    # ---------------- 1단계: probe 스캔 (지점별 독립 분류기) ----------------
    t0 = time.time()
    probe_rows = []
    for position in ("cls", "debt_span"):
        ha, hb = full_hidden[position]
        for layer in range(cc.N_HIDDEN_STATES):
            acc, std_dist = probe_heldout_acc(ha[:, layer, :].numpy(), hb[:, layer, :].numpy(),
                                               args.seed)
            probe_rows.append({"layer": layer, "position": position,
                               "probe_acc": acc, "standardized_distance": std_dist})
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(run_dir / f"rule{args.rule}_probe_scan.csv", index=False)
    print(f"probe 스캔 완료 ({time.time()-t0:.1f}s)")

    # ---------------- 2단계: activation patching ----------------
    # noisy 행(타 rule 탈락 문맥)은 rule 값과 무관하게 정답이 탈락이라 flip 실험 전제가 성립
    # 안 함 → patching은 clean 문맥 행으로만 (probe는 전체 사용 — v3b2의 "노이즈 속 판독" 검증)
    clean_pos = np.where(~noisy_mask)[0]
    n_patch = min(args.n_patch, len(clean_pos))
    patch_idx = clean_pos[:n_patch]
    df_a_sub = df_a.iloc[patch_idx].reset_index(drop=True)
    df_b_sub = df_b.iloc[patch_idx].reset_index(drop=True)
    infos_a = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in df_a_sub.iterrows()]
    infos_b = [cc.row_positions(tokenizer, r, rule_column, cc.MAX_LEN)[1] for _, r in df_b_sub.iterrows()]
    batch_a = build_batch(tokenizer, infos_a, device)
    batch_b = build_batch(tokenizer, infos_b, device)

    with torch.no_grad():
        clean_pred_a = model(**batch_a).logits.argmax(-1).cpu().numpy()
        clean_pred_b = model(**batch_b).logits.argmax(-1).cpu().numpy()
    valid_a = clean_pred_a == 1  # 통과로 정답 예측한 것만 p2f 대상
    valid_b = clean_pred_b == 0  # 탈락으로 정답 예측한 것만 f2p 대상
    print(f"clean 정답 예측: A={valid_a.sum()}/{n_patch}  B={valid_b.sum()}/{n_patch}")

    t0 = time.time()
    flip_rows = []
    for position in ("cls", "debt_span"):
        ha_full, hb_full = full_hidden[position]
        idxs_a = [idxs_for_position(info, position) for info in infos_a]
        idxs_b = [idxs_for_position(info, position) for info in infos_b]
        for layer in range(cc.N_HIDDEN_STATES):
            centroid_a = ha_full[:, layer, :].mean(0)
            centroid_b = hb_full[:, layer, :].mean(0)

            # centroid 주입 (그룹 평균 벡터)
            pred = run_with_hook(model, layer, batch_a,
                                  make_full_patch_hook(idxs_a, [centroid_b] * n_patch))
            flip_p2f_centroid = float((pred[valid_a] == 0).mean()) if valid_a.sum() else float("nan")
            pred = run_with_hook(model, layer, batch_b,
                                  make_full_patch_hook(idxs_b, [centroid_a] * n_patch))
            flip_f2p_centroid = float((pred[valid_b] == 1).mean()) if valid_b.sum() else float("nan")

            # paired 주입 (같은 문맥의 counterfactual 쌍 벡터 — patch_idx[i]의 상대는 patch_idx[i])
            donor_b_for_a = [hb_full[i, layer, :] for i in patch_idx]
            pred = run_with_hook(model, layer, batch_a, make_full_patch_hook(idxs_a, donor_b_for_a))
            flip_p2f_paired = float((pred[valid_a] == 0).mean()) if valid_a.sum() else float("nan")
            donor_a_for_b = [ha_full[i, layer, :] for i in patch_idx]
            pred = run_with_hook(model, layer, batch_b, make_full_patch_hook(idxs_b, donor_a_for_b))
            flip_f2p_paired = float((pred[valid_b] == 1).mean()) if valid_b.sum() else float("nan")

            flip_rows.append({
                "layer": layer, "position": position,
                "flip_p2f_centroid": flip_p2f_centroid, "flip_f2p_centroid": flip_f2p_centroid,
                "flip_p2f_paired": flip_p2f_paired, "flip_f2p_paired": flip_f2p_paired,
            })
            print(f"  L{layer:2d} {position:9s} | centroid p2f={flip_p2f_centroid:.3f} "
                  f"f2p={flip_f2p_centroid:.3f} | paired p2f={flip_p2f_paired:.3f} "
                  f"f2p={flip_f2p_paired:.3f}")
    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(run_dir / f"rule{args.rule}_flip_scan.csv", index=False)
    print(f"patching 완료 ({time.time()-t0:.1f}s)")

    # ---------------- 3단계: merge + 선별 ----------------
    merged = probe_df.merge(flip_df, on=["layer", "position"])
    merged.to_csv(run_dir / f"rule{args.rule}_scan_flip_merged.csv", index=False)

    cand_mask = (
        (merged["probe_acc"] >= PROBE_MIN)
        & (merged["flip_p2f_centroid"] >= FLIP_MIN)
        & (merged["flip_f2p_centroid"] >= FLIP_MIN)
    )
    diss_mask = (
        (merged["probe_acc"] >= PROBE_MIN)
        & (merged["flip_p2f_centroid"] < FLIP_DISS)
        & (merged["flip_f2p_centroid"] < FLIP_DISS)
    )
    candidates = merged[cand_mask].sort_values(
        ["position", "flip_p2f_centroid", "flip_f2p_centroid"], ascending=[True, False, False])
    dissociation = merged[diss_mask]

    l21_row = merged[(merged["layer"] == 21) & (merged["position"] == "debt_span")].iloc[0]
    summary = {
        "rule": args.rule, "context_mode": args.context_mode, "data": data_desc,
        "data_md5": data_md5, "n_pairs": int(len(df_a)), "n_patch": n_patch,
        "n_noisy": int(noisy_mask.sum()),
        "criteria": {"probe_min": PROBE_MIN, "flip_min": FLIP_MIN, "flip_dissociation": FLIP_DISS},
        "n_candidates": int(cand_mask.sum()),
        "candidates": candidates[["layer", "position", "probe_acc",
                                   "flip_p2f_centroid", "flip_f2p_centroid"]].to_dict("records"),
        "n_dissociation": int(diss_mask.sum()),
        "dissociation": dissociation[["layer", "position", "probe_acc",
                                       "flip_p2f_centroid", "flip_f2p_centroid"]].to_dict("records"),
        "L21_debt_span": {k: float(l21_row[k]) for k in
                          ("probe_acc", "flip_p2f_centroid", "flip_f2p_centroid",
                           "flip_p2f_paired", "flip_f2p_paired")},
        "L21_is_candidate": bool(((merged["layer"] == 21) & (merged["position"] == "debt_span")
                                   & cand_mask).any()),
    }
    with open(run_dir / f"rule{args.rule}_locate_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 플롯 ----------------
    for position in ("cls", "debt_span"):
        sub = merged[merged["position"] == position].sort_values("layer")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sub["layer"], sub["probe_acc"], marker="^", color="#27ae60", label="probe held-out acc")
        ax.plot(sub["layer"], sub["flip_p2f_centroid"], marker="o", color="#c0392b",
                label="flip 통과→탈락 (centroid)")
        ax.plot(sub["layer"], sub["flip_f2p_centroid"], marker="o", color="#2980b9",
                label="flip 탈락→통과 (centroid)")
        ax.plot(sub["layer"], sub["flip_p2f_paired"], marker="s", linestyle="--", color="#e67e22",
                label="flip 통과→탈락 (paired)")
        ax.plot(sub["layer"], sub["flip_f2p_paired"], marker="s", linestyle="--", color="#5dade2",
                label="flip 탈락→통과 (paired)")
        ax.axvline(21, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel("layer (0=embedding, 24=final)")
        ax.set_ylabel("accuracy / flip rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"rule{args.rule} locate (합성, {args.context_mode} 문맥) — {position}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / f"rule{args.rule}_locate_curve_{position}.png", dpi=150)
        plt.close(fig)

    cc.write_manifest(run_dir, script="centroid_locate_synth.py", inputs=vars(args),
                       extra={"data_md5": data_md5, "summary_file": f"rule{args.rule}_locate_report.json"},
                       start_time=start_time)
    print(f"\n후보 지점 {int(cand_mask.sum())}개 / 해리 {int(diss_mask.sum())}개")
    print(f"L21 x debt_span: {summary['L21_debt_span']} | 후보 포함 여부: {summary['L21_is_candidate']}")
    print(f"저장 완료: {run_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--context_mode", default="pairs", choices=CONTEXT_MODES,
                        help="locate 데이터 문맥 레시피 — edit(gate) 버전과 일치시켜 사용")
    parser.add_argument("--n_pairs", type=int, default=500)
    parser.add_argument("--n_patch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default=cc.HF_MODEL_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
