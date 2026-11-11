"""Phase 1: KoBBQ normalization + template splits, KMMLU-val, benign Korean prompts.

--eval_only builds only what scripts/08_final_eval.py needs (kobbq.parquet, kmmlu_dev.parquet).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse


from common import get_logger, load_config, save_json, seed_all
from data_benign import build_benign
from data_kobbq import load_kobbq, template_kfold, template_split
from eval_kmmlu import load_split, make_val
from prompts import assign_option_order

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--eval_only", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
seed_all(cfg["seed"])
P = cfg["paths"]
proc = Path(P["processed"]); proc.mkdir(parents=True, exist_ok=True)
log = get_logger("prep", Path(P["results"]) / "logs/01_prepare_data.log")

# ---- KoBBQ
kb = load_kobbq(P["kobbq_dir"], cfg["kobbq"]["unknown_text"])
split = template_split(kb, cfg["kobbq"]["split_ratio"], cfg["seed"])
folds = template_kfold(kb, cfg["kobbq"]["n_folds"], cfg["seed"])
save_json(split, Path(P["splits"]) / "kobbq_template_split.json")
save_json(folds, Path(P["splits"]) / "kobbq_5fold.json")
kb["split"] = kb.template_id.map(split)
kb["fold"] = kb.template_id.map(folds)
kb = assign_option_order(kb, cfg["seed"])
kb.to_parquet(proc / "kobbq.parquet")
log.info(f"KoBBQ samples={len(kb)} templates={kb.template_id.nunique()}")
log.info("templates per split x category:\n" +
         kb.drop_duplicates("template_id").pivot_table(index="category", columns="split", values="template_id",
                                                       aggfunc="count").to_string())
log.info("samples per split:\n" + kb.groupby(["split", "context_type"]).size().to_string())
for r in ["unknown", "stereo", "anti"]:
    log.info(f"{r} letter dist: {kb[f'{r}_option_idx'].value_counts().sort_index().to_dict()}")

# ---- KMMLU
dev = load_split(P["raw"], "dev")
dev.to_parquet(proc / "kmmlu_dev.parquet")
if args.eval_only:
    log.info(f"KMMLU dev={len(dev)}; --eval_only: skip KMMLU-val and benign prompts")
    sys.exit(0)
train = load_split(P["raw"], "train")
val = make_val(train, cfg["kmmlu"]["val_size"], cfg["seed"])
val.to_parquet(proc / "kmmlu_val.parquet")
log.info(f"KMMLU dev={len(dev)} train={len(train)} val(from train)={len(val)} subjects={val.subject.nunique()}")
log.info(f"KMMLU-val per group: {val.group.value_counts().to_dict()}")
# test is materialised only at Phase 7 (scripts/08_*); here we only read question text for dedup.
test_q = load_split(P["raw"], "test").question.astype(str)

# ---- benign
excl = set(test_q) | set(train.question.astype(str)) | set(dev.question.astype(str)) \
    | set(kb.context) | set(kb.question)
ben, stats = build_benign(cfg, excl, log.info)
ben.to_parquet(proc / "benign.parquet")
save_json(stats, Path(P["results"]) / cfg["run_name"] / "benign_stats.json")
log.info(f"benign total={len(ben)} split={ben.split.value_counts().to_dict()}")
assert len(ben) >= 14000, "benign < 14k: STOP and report"
