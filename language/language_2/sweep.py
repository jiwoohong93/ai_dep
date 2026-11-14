#!/usr/bin/env python3
"""논문 Table 5/6 replicate 전체 파이프라인 (resumable).

1) search: task x model x lr{1e-5,2e-5,3e-5,5e-5} x epoch{1..5}, seed 42 -> dev F1 로 최적 조합 선택
2) final : 선택된 조합으로 seed 5개 학습 -> test 점수 평균/표준편차
3) aggregate.py 로 결과 표 작성

metrics.json 이 이미 있는 run 은 건너뛰므로, 중간에 끊겨도 같은 명령으로 다시 실행하면 이어서 돈다.

실행 (세션이 끊겨도 계속 돌도록 nohup):
    cd ~/ai_dep_lang && nohup python language_2/sweep.py --gpus 0,1,2,3 --per_gpu 2 \
        > language_2/runs/sweep.log 2>&1 &
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PY = sys.executable

TASKS = ["A", "B", "C", "off_span", "tgt_span"]
MODELS = ["klue/bert-base", "klue/roberta-base", "klue/roberta-large"]
LRS = [1e-5, 2e-5, 3e-5, 5e-5]
EPOCHS = [1, 2, 3, 4, 5]
SEARCH_SEED = 42
FINAL_SEEDS = [1, 2, 3, 4, 5]


def short(model):
    return model.split("/")[-1]


def cost(job):
    """대략적인 소요 시간 (긴 것부터 돌려서 GPU 를 고르게 채운다)."""
    size = 3.0 if "large" in job["model"] else 1.0
    data = 2.0 if job["task"] == "A" else (0.6 if job["task"] == "C" else 1.0)
    return size * data * job["epochs"]


def run_jobs(jobs, gpus, per_gpu, log):
    def done(j):
        return (j["out"] / "metrics.json").exists() and (not j.get("save") or (j["out"] / "model").exists())
    jobs = sorted([j for j in jobs if not done(j)], key=cost, reverse=True)
    log(f"{len(jobs)} jobs to run")
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}  # slot index -> (proc, job, t0)
    failed = []
    while jobs or running:
        for si, g in enumerate(slots):
            if si not in running and jobs:
                j = jobs.pop(0)
                j["out"].mkdir(parents=True, exist_ok=True)
                cmd = [PY, str(HERE / "train.py"), "--task", j["task"], "--model", j["model"],
                       "--lr", str(j["lr"]), "--epochs", str(j["epochs"]), "--seed", str(j["seed"]),
                       "--out", str(j["out"])] + (["--save_model"] if j.get("save") else [])
                f = open(j["out"] / "train.log", "w")
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE.parent,
                                     env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": g,
                                          "TOKENIZERS_PARALLELISM": "false"})
                running[si] = (p, j, time.time())
        time.sleep(5)
        for si in list(running):
            p, j, t0 = running[si]
            if p.poll() is not None:
                del running[si]
                ok = p.returncode == 0 and done(j)
                rel = j["out"].relative_to(RUNS)
                if ok:
                    f1 = json.load(open(j["out"] / "metrics.json"))["dev"]["f1"]
                    log(f"done {rel} ({time.time() - t0:.0f}s) dev_f1={f1:.2f} | left {len(jobs)} queued, {len(running)} running")
                else:
                    failed.append(j)
                    log(f"FAILED {rel} rc={p.returncode} (see train.log)")
    return failed


def search_jobs():
    return [dict(task=t, model=m, lr=lr, epochs=ep, seed=SEARCH_SEED,
                 out=RUNS / "search" / t / short(m) / f"lr{lr:g}_ep{ep}")
            for t, m, lr, ep in itertools.product(TASKS, MODELS, LRS, EPOCHS)]


def select_best():
    best = {}
    for t, m in itertools.product(TASKS, MODELS):
        cands = []
        for lr, ep in itertools.product(LRS, EPOCHS):
            mp = RUNS / "search" / t / short(m) / f"lr{lr:g}_ep{ep}" / "metrics.json"
            if mp.exists():
                cands.append((json.load(open(mp))["dev"]["f1"], lr, ep))
        if cands:
            f1, lr, ep = max(cands)
            best[f"{t}/{short(m)}"] = {"task": t, "model": m, "lr": lr, "epochs": ep, "dev_f1": f1,
                                       "n_candidates": len(cands)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per_gpu", type=int, default=2)
    ap.add_argument("--stage", choices=["all", "search", "final"], default="all")
    args = ap.parse_args()
    gpus = args.gpus.split(",")
    RUNS.mkdir(exist_ok=True)

    def log(msg):
        print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)

    if args.stage in ("all", "search"):
        log("== search ==")
        failed = run_jobs(search_jobs(), gpus, args.per_gpu, log)
        if failed:
            log(f"{len(failed)} search jobs failed; final 단계는 실패한 조합을 제외하고 진행")

    if args.stage in ("all", "final"):
        best = select_best()
        json.dump(best, open(RUNS / "best_config.json", "w"), indent=2)
        log("best configs: " + json.dumps({k: (v["lr"], v["epochs"], round(v["dev_f1"], 2)) for k, v in best.items()}))
        log("== final ==")
        jobs = [dict(task=b["task"], model=b["model"], lr=b["lr"], epochs=b["epochs"], seed=s,
                     out=RUNS / "final" / b["task"] / short(b["model"]) / f"s{s}",
                     save=(s == FINAL_SEEDS[0]))  # 데모용으로 seed 하나만 모델 저장
                for b in best.values() for s in FINAL_SEEDS]
        run_jobs(jobs, gpus, args.per_gpu, log)

    subprocess.run([PY, str(HERE / "aggregate.py")], cwd=HERE.parent)
    log("== all done ==")


if __name__ == "__main__":
    main()
