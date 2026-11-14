#!/usr/bin/env python3
"""runs/ 결과를 모아 논문 Table 5/6 과 비교하는 표를 만든다 -> runs/results.md, runs/results.json

final 결과가 없으면 search 단계의 best(dev 기준) 조합의 seed 42 결과만 보여준다.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODELS = ["bert-base", "roberta-base", "roberta-large"]
TASK_NAMES = {"A": "Level A: Offensive", "B": "Level B: Target Type", "C": "Level C: Target Group",
              "off_span": "Offensive Span", "tgt_span": "Target Span"}
# 논문 (Jeong et al., 2022) Table 5, 6 의 (P, R, F1)
PAPER = {
    "A": {"bert-base": (75.8, 78.8, 77.2), "roberta-base": (76.8, 77.6, 77.2), "roberta-large": (76.6, 81.1, 78.8)},
    "B": {"bert-base": (58.5, 58.5, 57.6), "roberta-base": (60.8, 58.7, 59.4), "roberta-large": (63.9, 62.4, 62.7)},
    "C": {"bert-base": (56.9, 57.4, 55.9), "roberta-base": (58.6, 58.5, 57.3), "roberta-large": (59.1, 59.6, 57.5)},
    "off_span": {"bert-base": (47.7, 43.4, 41.9), "roberta-base": (46.4, 41.8, 40.6), "roberta-large": (50.8, 47.8, 45.4)},
    "tgt_span": {"bert-base": (60.2, 67.3, 61.9), "roberta-base": (60.6, 68.2, 62.4), "roberta-large": (60.8, 67.8, 62.5)},
}


def main():
    best = json.load(open(RUNS / "best_config.json")) if (RUNS / "best_config.json").exists() else {}
    out, lines = {}, []
    for table, tasks in (("Table 5 (category prediction)", ["A", "B", "C"]),
                         ("Table 6 (span prediction)", ["off_span", "tgt_span"])):
        span = tasks[0].endswith("_span")
        lines += [f"## {table}", "",
                  "| Task | Model | lr / ep | Ours P | Ours R | Ours F1 (mean ± std) | Paper F1 | Δ F1 | seeds |"
                  + (" F1, gold span 있는 샘플만 |" if span else ""),
                  "|---|---|---|---|---|---|---|---|---|" + ("---|" if span else "")]
        for t in tasks:
            for m in MODELS:
                b = best.get(f"{t}/{m}")
                files = sorted((RUNS / "final" / t / m).glob("s*/metrics.json"))
                if not files and b:  # final 전이면 search best 로 대체
                    files = [RUNS / "search" / t / m / f"lr{b['lr']:g}_ep{b['epochs']}" / "metrics.json"]
                files = [f for f in files if f.exists()]
                pp = PAPER[t][m]
                if not files:
                    lines.append(f"| {TASK_NAMES[t]} | {m} | - | - | - | - | {pp[2]} | - | 0 |" + (" - |" if t.endswith("_span") else ""))
                    continue
                full = [json.load(open(f)) for f in files]
                res = [d["test"] for d in full]
                ne = [d["test_nonempty_gold"]["f1"] for d in full if "test_nonempty_gold" in d]
                P, R, F = (np.array([r[k] for r in res]) for k in ("precision", "recall", "f1"))
                hp = f"{b['lr']:g} / {b['epochs']}" if b else "-"
                lines.append(f"| {TASK_NAMES[t]} | {m} | {hp} | {P.mean():.1f} | {R.mean():.1f} | "
                             f"{F.mean():.1f} ± {F.std(ddof=1) if len(F) > 1 else 0:.1f} | {pp[2]} | "
                             f"{F.mean() - pp[2]:+.1f} | {len(F)} |" + (f" {np.mean(ne):.1f} |" if t.endswith("_span") else ""))
                out[f"{t}/{m}"] = {"config": b, "n_seeds": len(F),
                                   "test": {"precision": P.mean(), "recall": R.mean(), "f1": F.mean(),
                                            "f1_std": F.std(ddof=1) if len(F) > 1 else 0.0},
                                   "paper": dict(zip(("precision", "recall", "f1"), pp)),
                                   **({"test_nonempty_gold_f1": float(np.mean(ne))} if ne else {})}
        lines.append("")
    md = ("# KOLD replication results (test set)\n\n"
          "Level C 는 논문과 같이 세부 target group 27-class. span 점수는 OFF 댓글 대상, 문자 단위 F1 의 샘플 평균"
          " (gold/pred 모두 비면 1, 한쪽만 비면 0). 마지막 열은 gold span 이 있는 샘플만으로 계산한 참고값.\n\n"
          + "\n".join(lines))
    (RUNS / "results.md").write_text(md, encoding="utf-8")
    json.dump(out, open(RUNS / "results.json", "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
