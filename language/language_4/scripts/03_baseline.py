"""Phase 2 baseline: KoBBQ train/val metrics (from the Phase-3 extraction pass) + KMMLU-val (prefix-cached).

Stops (exit 2) if the ambig baseline leaves no room: acc_amb >= 95% or anti errors dominate stereo errors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

import pandas as pd

from common import get_logger, load_config, load_model, run_dir, save_json, seed_all
from eval_kmmlu import accuracy, group_acc, run_kmmlu, shots_by_subject
from eval_kobbq import by_category, metrics, official_check
from extract import load_meta

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
seed_all(cfg["seed"])
P = cfg["paths"]
proc = Path(P["processed"])
R = run_dir(cfg, "results")
log = get_logger("baseline", Path(P["results"]) / "logs/03_baseline.log")

meta = load_meta(run_dir(cfg, "acts") / "kobbq_trainval")
out = {}
for sp in ("train", "val"):
    m = meta[meta.split == sp]
    out[f"kobbq_{sp}"] = metrics(m)
    out[f"kobbq_{sp}_official"] = official_check(m, Path(P["root"]) / "data/ext/KoBBQ/evaluation/5_evaluation.py")
    by_category(m).to_csv(R / f"baseline_kobbq_{sp}_by_category.csv")
    log.info(f"KoBBQ {sp}: {out[f'kobbq_{sp}']}\n official-code check: {out[f'kobbq_{sp}_official']}")
out["format"] = {"p_letters_mean": float(meta.p_letters.mean()), "p_letters_min": float(meta.p_letters.min()),
                 "p_space_letters_mean": float(meta.p_space_letters.mean())}

model, tok = load_model(cfg)
val = pd.read_parquet(proc / "kmmlu_val.parquet")
shots = shots_by_subject(pd.read_parquet(proc / "kmmlu_dev.parquet"), cfg["kmmlu"]["n_shot"])
kv = run_kmmlu(model, tok, val, shots, cfg, "base")
kv.drop(columns=["letter_logits"]).to_parquet(R / "baseline_kmmlu_val_preds.parquet")
out["kmmlu_val"] = {"acc": accuracy(kv), "by_group": group_acc(kv), "p_letters_mean": float(kv.p_letters.mean())}
log.info(f"KMMLU-val: {out['kmmlu_val']}")
save_json(out, R / "baseline.json")

tv = out["kobbq_val"]
if tv["acc_amb"] >= 0.95 or tv["amb_anti_rate"] >= tv["amb_stereo_rate"]:
    log.info("STOP: ambig baseline leaves little room for improvement (baseline ambig accuracy already near ceiling).")
    sys.exit(2)
