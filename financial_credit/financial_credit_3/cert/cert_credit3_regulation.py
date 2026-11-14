"""[금융/신용3] 공인시험용 평가 스크립트 — 정확도 / 재현율 / 응답시간

대상: Regulation 대출심사 모델(xlm-roberta-large-rule) + 정책 반영 편집
      (rule1 부채비율 기준 200% → 85% 강화, centroid v3b2 matched, gated shift)
데이터: Regulation/data/test/text/test.csv (은행×기업 38,160건)
정답(새 정책): 부채비율 외 8개 rule 통과 & 85% < 부채비율 ≤ 200% 인 '밴드' 기업 → 탈락(0),
             나머지는 기존 rule 라벨 그대로

화면 출력 항목
  1) 시험 데이터 SHA-256
  2) 정확도  = (TP+TN)/N                       목표 ≥ 0.95
  3) 재현율  = TP/(TP+FN) (통과=1 기준) + 탈락 기준·macro 병기    목표 ≥ 0.95
  4) 응답시간 = 1건 입력(문장 생성→토큰화→편집 hook 포함 추론→판정) 소요 시간
               평균 / 중앙값 / p95 / 최대 (초/건)               목표 ≤ 0.5 초/건

실행 (repo 루트에서):
  CUDA_VISIBLE_DEVICES=0 python cert/cert_credit3_regulation.py \
      --model_dir <xlm-roberta-large-rule 경로 또는 yongukpark/xlm-roberta-large-rule> \
      --shift_config Regulation/centroid/result/centroid_matched/edit_matched_v3b2/rule1_matched_shift_config.pt \
      --latency_n 1000
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Regulation" / "centroid" / "src" / "editing"))
import centroid_common as cc  # noqa: E402  (프롬프트·토큰 위치·rule 정의를 편집 코드와 동일하게 재사용)

RULE = 1
NEW_THRESHOLD = 85.0  # centroid_edit_matched.py와 동일


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GatedShift:
    """centroid_edit_shift.GatedShiftHook과 동일 로직: gate(w·h+b>0)가 켜진 샘플의 값 토큰에 alpha·δ_v 가산."""

    def __init__(self, idxs, cfg, alpha):
        self.idxs, self.cfg, self.alpha = idxs, cfg, alpha

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        w = self.cfg["w_gate"].to(h.dtype).to(h.device)
        d = (self.alpha * self.cfg["delta_v"]).to(h.dtype).to(h.device)
        for i, idx in enumerate(self.idxs):
            if (torch.dot(h[i, idx, :].mean(0), w) + self.cfg["b_gate"]).item() > 0:
                h[i, idx, :] = h[i, idx, :] + d
        return (h,) + tuple(output[1:]) if is_tuple else h


class Judge:
    """1건/배치 판정기 — 응답시간 측정 시 이 호출 전체를 잰다."""

    def __init__(self, model, tokenizer, cfg, alpha, rule_column, device):
        self.model, self.tok, self.cfg, self.alpha = model, tokenizer, cfg, alpha
        self.rule_column, self.device = rule_column, device
        self.module = model.roberta.encoder.layer[cfg["layer"] - 1] if cfg is not None else None

    @torch.no_grad()
    def __call__(self, rows: pd.DataFrame):
        infos = [cc.row_positions(self.tok, r, self.rule_column, cc.MAX_LEN)[1] for _, r in rows.iterrows()]
        enc = self.tok.pad({"input_ids": [x["input_ids"] for x in infos]}, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        handle = None
        if self.cfg is not None:
            handle = self.module.register_forward_hook(
                GatedShift([x["value_all"] for x in infos], self.cfg, self.alpha))
        try:
            logits = self.model(**enc).logits
        finally:
            if handle is not None:
                handle.remove()
        return logits.argmax(-1).cpu().numpy()


def new_policy_labels(df, rule_column, theta):
    decisive = cc.other_rules_pass_mask(df, rule_column)
    v = pd.to_numeric(df[rule_column], errors="coerce")
    band = decisive & (v > NEW_THRESHOLD) & (v <= theta)
    y = df["label"].copy()
    y.loc[band] = 0
    return y.to_numpy(), band.to_numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=cc.LOCAL_MODEL_DIR)
    p.add_argument("--test_csv", default=str(cc.TEST_CSV))
    p.add_argument("--shift_config", default=str(cc.CENTROID_ROOT / "result" / "centroid_matched"
                                                 / "edit_matched_v3b2" / "rule1_matched_shift_config.pt"))
    p.add_argument("--alpha", type=float, default=None, help="기본: shift_config의 best_alpha")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--latency_n", type=int, default=1000, help="응답시간 측정 건수 (0=생략, -1=전체)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--save_json", default=None)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _, rule_column, theta, *_ = cc.rule_spec(RULE)
    test_path = Path(args.test_csv)
    print(f"시험 데이터: {test_path}\nSHA-256   : {sha256(test_path)}")
    df = pd.read_csv(test_path, low_memory=False).dropna().reset_index(drop=True)
    y_new, band = new_policy_labels(df, rule_column, theta)
    print(f"시험 데이터 수 {len(df)} / 정책 변경으로 라벨이 바뀐 밴드 기업 행 {int(band.sum())}")

    t0 = time.time()
    tokenizer, model = cc.load_model(args.model_dir, device=device)
    cfg = torch.load(args.shift_config)
    alpha = args.alpha if args.alpha is not None else float(cfg["best_alpha"])
    judge = Judge(model, tokenizer, cfg, alpha, rule_column, device)
    print(f"모델 로딩 완료 ({time.time()-t0:.1f}s, device={device}, layer={cfg['layer']}, alpha={alpha})")
    report = {"test_sha256": sha256(test_path), "alpha": alpha, "layer": int(cfg["layer"])}

    if not args.skip_accuracy:
        preds = np.concatenate([judge(df.iloc[i:i + args.batch_size])
                                for i in range(0, len(df), args.batch_size)])
        r = {
            "accuracy": float(accuracy_score(y_new, preds)),
            "recall_pass": float(recall_score(y_new, preds, pos_label=1)),
            "recall_fail": float(recall_score(y_new, preds, pos_label=0)),
            "macro_recall": float(recall_score(y_new, preds, average="macro")),
            "policy_reflection_band": float((preds[band] == 0).mean()),
        }
        report["accuracy"] = r
        print("\n===== 정확도 / 재현율 (새 정책 라벨 기준) =====")
        print(f"  정확도                  : {r['accuracy']:.4f}   (목표 ≥ 0.95)")
        print(f"  재현율 (통과=1)         : {r['recall_pass']:.4f}   (목표 ≥ 0.95)")
        print(f"  재현율 (탈락=0)         : {r['recall_fail']:.4f}")
        print(f"  재현율 (macro)          : {r['macro_recall']:.4f}")
        print(f"  (참고) 밴드 정책 반영률 : {r['policy_reflection_band']:.4f}")

    if args.latency_n != 0:
        n = len(df) if args.latency_n < 0 else min(args.latency_n, len(df))
        sample = df.sample(n=n, random_state=0) if n < len(df) else df
        sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
        for i in range(args.warmup):
            judge(sample.iloc[[i % n]])
        sync()
        times = []
        for i in range(n):
            t = time.perf_counter()
            judge(sample.iloc[[i]])
            sync()
            times.append(time.perf_counter() - t)
        times_sorted = sorted(times)
        lat = {"n": n, "mean_s": statistics.mean(times), "median_s": statistics.median(times),
               "p95_s": times_sorted[int(0.95 * (n - 1))], "max_s": times_sorted[-1],
               "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"}
        report["latency"] = lat
        print(f"\n===== 응답시간 (1건씩 입력, {n}건, 장치: {lat['device']}) =====")
        print(f"  평균   : {lat['mean_s']:.4f} 초/건   (목표 ≤ 0.5)")
        print(f"  중앙값 : {lat['median_s']:.4f} 초/건")
        print(f"  p95    : {lat['p95_s']:.4f} 초/건")
        print(f"  최대   : {lat['max_s']:.4f} 초/건")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.save_json, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
