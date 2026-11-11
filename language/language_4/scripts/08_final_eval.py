"""Phase 7: one-shot test evaluation of the frozen selections (KoBBQ test templates, full KMMLU test).

Conditions: base, alphasteer (selected) by default; prompt, caa (selected) and alphasteer_alt (the alphasteer
variant with the other r kind, when selectable) on request. caa needs directions.pt from scripts/04_directions.py.
Resumable per condition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import json

import pandas as pd

from common import get_logger, load_config, load_model, run_dir, save_json, seed_all
from eval_kmmlu import load_split, run_kmmlu, shots_by_subject
from eval_kobbq import by_category, metrics, official_check
from hooks import Steerer
from prompts import kobbq_prompt
from scoring import score_texts
from steer_eval import build_spec

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--hb_tag", default="alphasteer")
ap.add_argument("--conditions", default="base,alphasteer")
args = ap.parse_args()
cfg = load_config(args.config)
seed_all(cfg["seed"])
P = cfg["paths"]
proc = Path(P["processed"])
S, R = run_dir(cfg, "steer"), run_dir(cfg, "results")
T = R / "test"
T.mkdir(exist_ok=True)
log = get_logger("final", Path(P["results"]) / "logs/08_final_eval.log")

sel = json.load(open(R / f"selected_{args.hb_tag}.json"))["selected"]
save_json(sel, T / "frozen_selection.json")
alt_kind = None
if sel.get("alphasteer"):
    alt_kind = "caa" if sel["alphasteer"]["r_kind"] == "behavior" else "behavior"
conds = {"base": None, "prompt": None, "caa": sel.get("caa"), "alphasteer": sel.get("alphasteer"),
         "alphasteer_alt": sel.get(f"alphasteer_{alt_kind}") if alt_kind else None}

kb = pd.read_parquet(proc / "kobbq.parquet")
kb_test = kb[kb.split == "test"].reset_index(drop=True)
km_test = load_split(P["raw"], "test")
shots = shots_by_subject(pd.read_parquet(proc / "kmmlu_dev.parquet"), cfg["kmmlu"]["n_shot"])
log.info(f"TEST: KoBBQ test n={len(kb_test)} ({kb_test.template_id.nunique()} templates), KMMLU test n={len(km_test)}")
model, tok = load_model(cfg)

for name in args.conditions.split(","):
    if name not in ("base", "prompt") and not conds.get(name):
        log.info(f"skip {name}: nothing selected")
        continue
    f_kb, f_km = T / f"kobbq_{name}.parquet", T / f"kmmlu_{name}.parquet"
    if f_kb.exists() and f_km.exists():
        log.info(f"skip {name}: done")
        continue
    c = conds[name] or {}
    cond = "alphasteer" if name.startswith("alphasteer") else name
    spec = build_spec(cond, S, c.get("layers", []), c.get("r_kind", ""), c.get("p", 0), c.get("alpha", 0), args.hb_tag)
    lam = c.get("lam", 0.0)
    texts = [kobbq_prompt(tok, r, cfg, "prompt" if cond == "prompt" else "base") for r in kb_test.to_dict("records")]
    with Steerer(model, spec, lam, cfg["alphasteer"]["mode"]):
        r = score_texts(model, tok, texts, "ABC", cfg["extract"]["max_tokens_per_batch"])
    kbp = kb_test.copy()
    kbp["pred_idx"], kbp["margin"] = r["pred_idx"], r["margin"]
    kbp["letter_logits"] = list(r["letter_logits"])
    kbp.to_parquet(f_kb)
    log.info(f"{name} KoBBQ test: {metrics(kbp)}")
    with Steerer(model, spec, lam, cfg["alphasteer"]["mode"]):
        kmp = run_kmmlu(model, tok, km_test, shots, cfg, "prompt" if cond == "prompt" else "base",
                        cfg["extract"]["max_tokens_per_batch"])
    kmp.to_parquet(f_km)
    log.info(f"{name} KMMLU test acc={(kmp.pred_idx == kmp.answer - 1).mean():.4f}")

# ---- final table
rows, cats = [], []
km_base = None
for name in ["base", "prompt", "caa", "alphasteer", "alphasteer_alt"]:
    f_kb, f_km = T / f"kobbq_{name}.parquet", T / f"kmmlu_{name}.parquet"
    if not (f_kb.exists() and f_km.exists()):
        continue
    kbp, kmp = pd.read_parquet(f_kb), pd.read_parquet(f_km)
    m = metrics(kbp)
    off = official_check(kbp, Path(P["root"]) / "data/ext/KoBBQ/evaluation/5_evaluation.py")
    for k in ("acc_amb", "acc_dis", "bias_amb", "bias_dis"):
        assert abs(off[k] - m[k]) < 1e-9, (name, k, off[k], m[k])
    acc = float((kmp.pred_idx == kmp.answer - 1).mean())
    km_base = acc if name == "base" else km_base
    grp = kmp.assign(c=kmp.pred_idx == kmp.answer - 1).groupby("group").c.mean()
    c = conds.get(name) or {}
    rows.append({"condition": name, "r_kind": c.get("r_kind", ""), "layers": str(c.get("layers", "")),
                 "p": c.get("p", ""), "alpha": c.get("alpha", ""), "lambda": c.get("lam", ""),
                 "kobbq_amb_acc": m["acc_amb"], "kobbq_dis_acc": m["acc_dis"], "bias_amb": m["bias_amb"],
                 "bias_dis": m["bias_dis"], "amb_stereo_rate": m["amb_stereo_rate"], "amb_anti_rate": m["amb_anti_rate"],
                 "amb_anti_share_of_errors": m["amb_anti_share_of_errors"], "kmmlu_acc": acc,
                 "rel_drop": (km_base - acc) / km_base * 100, "abs_drop": (km_base - acc) * 100,
                 **{f"kmmlu_{g}": v for g, v in grp.items()}})
    bc = by_category(kbp)
    bc["condition"] = name
    cats.append(bc.reset_index().rename(columns={"index": "category"}))
table = pd.DataFrame(rows)
base_grp = table.iloc[0]
for g in [c for c in table.columns if c.startswith("kmmlu_") and c not in ("kmmlu_acc",)]:
    table[g.replace("kmmlu_", "rel_drop_")] = (base_grp[g] - table[g]) / base_grp[g] * 100
table.to_csv(Path(P["results"]) / "final_table.csv", index=False)
pd.concat(cats).to_csv(Path(P["results"]) / "final_by_category.csv", index=False)
log.info("\n" + table.round(4).to_string())
