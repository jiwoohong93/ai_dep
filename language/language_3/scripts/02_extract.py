"""Phase 3 (+ Phase 2 base predictions): last-prompt-token and CAA answer-token activations.

Sets (all layers 0..L-1, decoder-layer outputs):
  kobbq_trainval : every KoBBQ train/val sample, base prompt   -> also the Phase-2 KoBBQ baseline
  benign         : all benign prompts (train + val)
  kmmlu_val      : KMMLU-val 5-shot prompts (full prompt, no prefix cache)
  caa_unknown / caa_stereo : KoBBQ train-ambig prompts + assistant letter of the unknown / stereo option
KoBBQ test activations are extracted only in Phase 7 (scripts/08_final_eval.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

import pandas as pd

from common import get_logger, load_config, load_model, run_dir, seed_all
from eval_kmmlu import shots_by_subject
from extract import extract
from prompts import LETTERS, benign_prompt, kmmlu_prompt, kobbq_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--sets", default="kobbq_trainval,caa_unknown,caa_stereo,benign,kmmlu_val")
args = ap.parse_args()
cfg = load_config(args.config)
seed_all(cfg["seed"])
P = cfg["paths"]
proc = Path(P["processed"])
log = get_logger("extract", Path(P["results"]) / "logs/02_extract.log")
model, tok = load_model(cfg)
L = model.config.num_hidden_layers
layers = list(range(L))
A = run_dir(cfg, "acts")
E = cfg["extract"]

kb = pd.read_parquet(proc / "kobbq.parquet")
assert "test" not in set(kb[kb.split != "test"].split)
for name in args.sets.split(","):
    if (A / name / "DONE").exists():
        log.info(f"skip {name} (DONE)")
        continue
    if name == "kobbq_trainval":
        meta = kb[kb.split.isin(["train", "val"])].reset_index(drop=True)
        texts = [kobbq_prompt(tok, r, cfg) for r in meta.to_dict("records")]
        letters = "ABC"
    elif name in ("caa_unknown", "caa_stereo"):
        meta = kb[(kb.split == "train") & (kb.context_type == "ambig")].reset_index(drop=True)
        role = name.split("_")[1]
        texts = [kobbq_prompt(tok, r, cfg, answer_letter=LETTERS[r[f"{role}_option_idx"]])
                 for r in meta.to_dict("records")]
        letters = "ABC"
    elif name == "benign":
        meta = pd.read_parquet(proc / "benign.parquet")
        texts = [benign_prompt(tok, t, cfg) for t in meta.text]
        letters = "ABC"  # scoring irrelevant here
    elif name == "kmmlu_val":
        meta = pd.read_parquet(proc / "kmmlu_val.parquet")
        shots = shots_by_subject(pd.read_parquet(proc / "kmmlu_dev.parquet"), cfg["kmmlu"]["n_shot"])
        texts = [kmmlu_prompt(tok, r, shots[r["subject"]], cfg) for r in meta.to_dict("records")]
        letters = "ABCD"
    else:
        raise ValueError(name)
    if name.startswith("caa"):
        # sanity: the captured position is the appended letter token
        ids = tok(texts[0], add_special_tokens=False).input_ids
        assert tok.decode(ids[-1]) in "ABC", tok.decode(ids[-3:])
    log.info(f"{name}: example prompt\n{texts[0]}")
    extract(model, tok, texts, meta, A / name, letters, layers, E["shard_size"], E["max_tokens_per_batch"], log.info)
