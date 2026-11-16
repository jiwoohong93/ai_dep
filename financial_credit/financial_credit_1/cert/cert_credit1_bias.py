"""[금융/신용1] 공인시험용 평가 스크립트 — 예측정확도 / 공정성(DEO)

대상: Bias(UCI Adult) cgmix_rel 모델 + 비혼 페널티 완화 편집(gated centroid shift, layer 9)
데이터: Bias/data/test/text/test.csv (UCI 공식 adult.test 16,281건, 학습·편집 구성에 미사용)

화면 출력 항목
  1) 시험 데이터 SHA-256
  2) 예측정확도  = (TP+TN)/N                                   목표 ≥ 0.82
  3) 공정성 DEO  = | TPR_male − TPR_female |                    목표 ≤ 0.05
       TPR_g = P(ŷ=1 | y=1, sex=g)  (Hardt et al. 2016 Equal Opportunity)
     (참고) 그룹 정확도 격차 |Acc_male − Acc_female| 도 함께 출력

실행 (repo 루트에서):
  CUDA_VISIBLE_DEVICES=0 python cert/cert_credit1_bias.py \
      --shift_config Bias/result/bias_centroid/edit/layer9_seed0/shift_config.pt --alpha 0.9

--no_edit 를 주면 편집 전(base) 모델만 평가한다.
--married_config/--beta 는 기혼 쪽 과승인 교정(양방향 편집) 실험용 옵션.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Bias" / "src" / "editing"))
import bias_centroid_common as bc  # noqa: E402  (프롬프트·토큰 위치 계산을 학습/편집 코드와 동일하게 재사용)

UNMARRIED = ["Never-married", "Divorced", "Separated", "Widowed"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TwoSidedGatedShift:
    """gate score>0(비혼으로 읽힘) → +alpha·δ_v (비혼 과소승인 완화, 기존 편집과 동일)
    gate score≤0(기혼으로 읽힘) → −beta·δ_v (기혼 과승인 완화, beta=0이면 기존 편집과 완전 동일)"""

    def __init__(self, idxs_m, idxs_r, cfg, alpha, beta=0.0, cfg_married=None):
        self.idxs_m, self.idxs_r = idxs_m, idxs_r
        self.cfg, self.alpha, self.beta = cfg, alpha, beta
        self.cfg_married = cfg_married or cfg

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        w = self.cfg["w_gate"].to(h.dtype).to(h.device)
        b = self.cfg["b_gate"]
        dm = (self.alpha * self.cfg["delta_v_marital"]).to(h.dtype).to(h.device)
        dr = (self.alpha * self.cfg["delta_v_relationship"]).to(h.dtype).to(h.device)
        dm2 = (self.beta * self.cfg_married["delta_v_marital"]).to(h.dtype).to(h.device)
        dr2 = (self.beta * self.cfg_married["delta_v_relationship"]).to(h.dtype).to(h.device)
        for i in range(h.shape[0]):
            feat = torch.cat([h[i, self.idxs_m[i], :].mean(0), h[i, self.idxs_r[i], :].mean(0)])
            score = (torch.dot(feat, w) + b).item()
            if score > 0 and self.alpha != 0:
                h[i, self.idxs_m[i], :] += dm
                h[i, self.idxs_r[i], :] += dr
            elif score <= 0 and self.beta != 0:
                h[i, self.idxs_m[i], :] -= dm2
                h[i, self.idxs_r[i], :] -= dr2
        return (h,) + tuple(output[1:]) if is_tuple else h


@torch.no_grad()
def predict(model, tokenizer, df, device, module=None, hook_kwargs=None, batch_size=64):
    preds = []
    for i in range(0, len(df), batch_size):
        sub = df.iloc[i:i + batch_size]
        infos = [bc.row_positions(tokenizer, r)[1] for _, r in sub.iterrows()]
        enc = tokenizer.pad({"input_ids": [x["input_ids"] for x in infos]}, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        handle = None
        if hook_kwargs is not None:
            hook = TwoSidedGatedShift([x["marital-status"] for x in infos],
                                      [x["relationship"] for x in infos], **hook_kwargs)
            handle = module.register_forward_hook(hook)
        try:
            out = model(**enc)
        finally:
            if handle is not None:
                handle.remove()
        preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def metrics(df, preds):
    y = df["label"].to_numpy()
    male = (df["sex"] == "Male").to_numpy()
    unm = df["marital-status"].isin(UNMARRIED).to_numpy()
    tpr = lambda m: float((preds[m & (y == 1)] == 1).mean())
    acc = lambda m: float((preds[m] == y[m]).mean())
    out = {
        "n": int(len(y)), "accuracy": acc(np.ones_like(male)),
        "tpr_male": tpr(male), "tpr_female": tpr(~male),
        "acc_male": acc(male), "acc_female": acc(~male),
        "fpr_male": float((preds[male & (y == 0)] == 1).mean()),
        "fpr_female": float((preds[~male & (y == 0)] == 1).mean()),
        "recall_unmarried": tpr(unm), "recall_married": tpr(~unm),
    }
    out["deo"] = abs(out["tpr_male"] - out["tpr_female"])
    out["group_acc_gap"] = abs(out["acc_male"] - out["acc_female"])
    return out


def show(tag, m):
    print(f"\n===== [{tag}] =====")
    print(f"  시험 데이터 수                 : {m['n']}")
    print(f"  예측정확도 (Accuracy)          : {m['accuracy']:.4f}   (목표 ≥ 0.82)")
    print(f"  TPR 남성 / 여성                : {m['tpr_male']:.4f} / {m['tpr_female']:.4f}")
    print(f"  공정성 DEO = |TPR_m − TPR_f|   : {m['deo']:.4f}   (목표 ≤ 0.05)")
    print(f"  (참고) 그룹 정확도 격차        : {m['group_acc_gap']:.4f}  "
          f"[Acc_m {m['acc_male']:.4f} / Acc_f {m['acc_female']:.4f}]")
    print(f"  (참고) FPR 남성 / 여성         : {m['fpr_male']:.4f} / {m['fpr_female']:.4f}")
    print(f"  (참고) recall 비혼 / 기혼      : {m['recall_unmarried']:.4f} / {m['recall_married']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=str(bc.MODEL_DIR))
    p.add_argument("--test_csv", default=str(bc.TEST_CSV))
    p.add_argument("--shift_config", default=str(bc.RESULT_ROOT / "edit" / "layer9_seed0" / "shift_config.pt"))
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.0, help="기혼 쪽 반대방향 shift 크기 (실험용, 기본 0)")
    p.add_argument("--married_config", default=None, help="기혼 쪽에 다른 δ_v를 쓸 때 (기본: shift_config 재사용)")
    p.add_argument("--no_edit", action="store_true")
    p.add_argument("--with_base", action="store_true", help="편집 전 결과도 같이 출력")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_json", default=None)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_path = Path(args.test_csv)
    print(f"시험 데이터: {test_path}\nSHA-256   : {sha256(test_path)}")
    df = pd.read_csv(test_path).dropna().reset_index(drop=True)

    t0 = time.time()
    tokenizer, model = bc.load_model(args.model_dir, device=device)
    print(f"모델 로딩 완료: {args.model_dir} ({time.time()-t0:.1f}s, device={device})")

    report = {"test_sha256": sha256(test_path), "alpha": args.alpha, "beta": args.beta}
    if args.no_edit or args.with_base:
        report["base"] = metrics(df, predict(model, tokenizer, df, device, batch_size=args.batch_size))
        show("편집 전 (base)", report["base"])
    if not args.no_edit:
        cfg = torch.load(args.shift_config)
        cfg_m = torch.load(args.married_config) if args.married_config else None
        module = model.roberta.encoder.layer[cfg["layer"] - 1]
        hk = dict(cfg=cfg, alpha=args.alpha, beta=args.beta, cfg_married=cfg_m)
        report["edited"] = metrics(df, predict(model, tokenizer, df, device, module, hk, args.batch_size))
        show(f"편집 후 (layer {cfg['layer']}, alpha={args.alpha}, beta={args.beta})", report["edited"])
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.save_json, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
