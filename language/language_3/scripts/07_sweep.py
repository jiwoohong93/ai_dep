"""Phase 6b: hyper-parameter sweep on validation data only, then the pre-fixed selection rule.

Stages
  screen1 : base, prompt; alphasteer (r in {behavior, caa}) x layer candidates x lambda (p, alpha default);
            caa (r_caa) x layer candidates x rho.
  screen2 : top-2 (r, layers) alphasteer groups from screen1 x p x alpha x lambda.
  full    : configs feasible on screening (KMMLU-val rel drop <= 2.0, screen disamb drop <= 3.0pp) -> top-6 per
            (condition, r) by screen ambig gain, + base/prompt, re-scored on the FULL KoBBQ val.
  select  : selection rule on full-val results (KMMLU-val rel drop <= 2.0% AND disamb drop <= 2.0pp, max ambig
            gain, tie -> smaller lambda).
KoBBQ screening set = <= 8 samples per (template, version, context, polarity) cell of val (2,360 samples).
KMMLU-val (2,000) is always evaluated in full, except for steering configs that cannot be selected anyway
(screen ambig acc <= base, or screen disamb drop > 3pp) -> recorded with kmmlu NaN. lambda chains run ascending and
stop once KMMLU rel drop > 3% or disamb drop > 5pp. Every worker evaluates `base` first (same key -> written once
per worker, identical results).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import hashlib
import json
import time

import pandas as pd

from common import get_logger, load_config, run_dir, save_json, seed_all

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--stage", required=True, choices=["screen1", "screen2", "full", "select"])
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--hb_tag", default="alphasteer")
ap.add_argument("--top_full", type=int, default=6)
args = ap.parse_args()
cfg = load_config(args.config)
seed_all(cfg["seed"])
P = cfg["paths"]
proc = Path(P["processed"])
S, R = run_dir(cfg, "steer"), run_dir(cfg, "results")
SW = R / "sweep"
SW.mkdir(exist_ok=True)
log = get_logger(f"sweep{args.worker}", Path(P["results"]) / f"logs/07_sweep_{args.stage}_w{args.worker}.log")
st, ac = cfg["steer"], cfg["alphasteer"]
STOP_REL = 3.0


def key(c):
    return json.dumps({k: c[k] for k in sorted(c)}, sort_keys=True)


def load_results(stage):
    rows = []
    for f in SW.glob(f"{stage}_w*.jsonl"):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def feasible(df, base, dis_tol):
    return df[(df.kmmlu_rel_drop_pct <= st["select"]["max_kmmlu_rel_drop"]) &
              ((base["acc_dis"] - df.acc_dis) * 100 <= dis_tol)]


def chains_screen1():
    cands = json.load(open(R / "layer_candidates.json"))
    ch = [[dict(cond="prompt", layers=[], r_kind="", p=0.0, alpha=0.0, lam=0.0)]]
    for rk in ("behavior", "caa"):
        for ls in cands[rk]:
            ch.append([dict(cond="alphasteer", layers=ls, r_kind=rk, p=ac["null_ratio"], alpha=ac["ridge"], lam=x)
                       for x in st["lambdas"]])
    for ls in cands["caa"]:
        ch.append([dict(cond="caa", layers=ls, r_kind="caa", p=0.0, alpha=0.0, lam=x) for x in st["caa_rhos"]])
    return ch


def chains_screen2():
    s1 = load_results("screen1")
    base = s1[s1.cond == "base"].iloc[0]
    a = feasible(s1[s1.cond == "alphasteer"], base, 3.0).copy()
    a["gkey"] = a.r_kind + "|" + a.layers.map(str)
    top = a.sort_values("acc_amb", ascending=False).drop_duplicates("gkey").head(2)
    log.info(f"screen2 groups: {top[['r_kind', 'layers', 'lam', 'acc_amb']].to_dict('records')}")
    ch = []
    for _, t in top.iterrows():
        for p in ac["sweep_null_ratio"]:
            for al in ac["sweep_ridge"]:
                if p == ac["null_ratio"] and al == ac["ridge"]:
                    continue
                ch.append([dict(cond="alphasteer", layers=list(t.layers), r_kind=t.r_kind, p=p, alpha=al, lam=x)
                           for x in st["lambdas"]])
    return ch


def chains_full():
    s = pd.concat([load_results("screen1"), load_results("screen2")], ignore_index=True)
    base = s[s.cond == "base"].iloc[0]
    f = feasible(s[s.cond.isin(["alphasteer", "caa"])], base, 3.0)
    picks = [s[s.cond.isin(["base", "prompt"])]]
    for (c, rk), g in f.groupby(["cond", "r_kind"]):
        picks.append(g.sort_values(["acc_amb", "lam"], ascending=[False, True]).head(args.top_full))
    sel = pd.concat(picks)
    cols = ["cond", "layers", "r_kind", "p", "alpha", "lam"]
    # every screen1 worker wrote its own `base` row -> evaluate each config once
    sel = sel[~sel[cols].astype(str).duplicated()]
    log.info(f"full-val candidates:\n{sel[cols + ['acc_amb', 'acc_dis', 'kmmlu_rel_drop_pct']].to_string()}")
    out = []
    for r in sel.to_dict("records"):
        c = {k: r[k] for k in cols}
        c["layers"] = list(c["layers"])
        c["kmmlu_from_screen"] = {k: r[k] for k in r if k.startswith("kmmlu_")}
        out.append([c])
    return out


if args.stage == "select":
    full = load_results("full")
    base = full[full.cond == "base"].iloc[0]
    full["amb_gain_pp"] = (full.acc_amb - base.acc_amb) * 100
    full["dis_drop_pp"] = (base.acc_dis - full.acc_dis) * 100
    full.to_csv(R / "sweep_full_val.csv", index=False)
    fe = feasible(full, base, st["select"]["max_disamb_drop"])
    sel = {}
    for name, g in [("alphasteer", fe[fe.cond == "alphasteer"]),
                    ("alphasteer_behavior", fe[(fe.cond == "alphasteer") & (fe.r_kind == "behavior")]),
                    ("alphasteer_caa", fe[(fe.cond == "alphasteer") & (fe.r_kind == "caa")]),
                    ("caa", fe[fe.cond == "caa"])]:
        if len(g):
            best = g.sort_values(["amb_gain_pp", "lam"], ascending=[False, True]).iloc[0]
            sel[name] = {k: (list(v) if k == "layers" else v) for k, v in best.items()
                         if k in ["cond", "layers", "r_kind", "p", "alpha", "lam", "acc_amb", "acc_dis", "bias_amb",
                                  "bias_dis", "kmmlu_acc", "kmmlu_rel_drop_pct", "amb_gain_pp", "dis_drop_pp"]}
        else:
            sel[name] = None
    sel["base"] = {k: base[k] for k in ["acc_amb", "acc_dis", "bias_amb", "bias_dis", "kmmlu_acc"]}
    save_json({"hb_tag": args.hb_tag, "selected": sel}, R / f"selected_{args.hb_tag}.json")
    log.info(json.dumps(sel, indent=1, default=float))
    sys.exit(0)

# ---------------- evaluation stages

from common import load_model  # noqa: E402
from eval_kmmlu import shots_by_subject  # noqa: E402
from eval_kobbq import metrics  # noqa: E402
from steer_eval import build_spec, eval_kmmlu, eval_kobbq, screen_subset, summarize  # noqa: E402

chains = {"screen1": chains_screen1, "screen2": chains_screen2, "full": chains_full}[args.stage]()
mine = chains[args.worker::args.n_workers]
out_f = SW / f"{args.stage}_w{args.worker}.jsonl"
done = {}
for f in SW.glob(f"{args.stage}_w*.jsonl"):
    done |= {(r := json.loads(l))["key"]: r for l in open(f)}
log.info(f"{args.stage}: {len(chains)} chains total, {len(mine)} for worker {args.worker}, {len(done)} configs done")

kb = pd.read_parquet(proc / "kobbq.parquet")
kb_val = kb[kb.split == "val"].reset_index(drop=True)
assert set(kb_val.split) == {"val"}
kb_eval = kb_val if args.stage == "full" else screen_subset(kb_val)
kv = pd.read_parquet(proc / "kmmlu_val.parquet")
shots = shots_by_subject(pd.read_parquet(proc / "kmmlu_dev.parquet"), cfg["kmmlu"]["n_shot"])
km_base = json.load(open(R / "baseline.json"))["kmmlu_val"]["acc"]
log.info(f"KoBBQ eval n={len(kb_eval)} | KMMLU-val n={len(kv)} base acc={km_base:.4f}")
model, tok = load_model(cfg)
base_kb = None
prev = load_results("screen1" if args.stage == "screen2" else args.stage)
if len(prev) and (prev.cond == "base").any():
    base_kb = prev[prev.cond == "base"].iloc[0].to_dict()
elif args.stage != "full":
    mine = [[dict(cond="base", layers=[], r_kind="", p=0.0, alpha=0.0, lam=0.0)]] + \
           [ch for ch in mine if ch[0]["cond"] != "base"]
PRED = SW / "preds"
PRED.mkdir(exist_ok=True)

for chain in mine:
    for c in chain:
        km_cached = c.pop("kmmlu_from_screen", None)
        c["hb_tag"] = args.hb_tag
        k = key(c)
        if k in done:
            # resumed: apply the same chain-stop rule to the recorded result
            m = done[k]
            if m["kmmlu_rel_drop_pct"] > STOP_REL or (base_kb and (base_kb["acc_dis"] - m["acc_dis"]) * 100 > 5.0):
                break
            continue
        t0 = time.time()
        spec = build_spec(c["cond"], S, c["layers"], c["r_kind"], c["p"], c["alpha"], args.hb_tag)
        kbp = eval_kobbq(model, tok, kb_eval, cfg, c["cond"], spec, c["lam"])
        mk = metrics(kbp)
        skip = None
        if km_cached is not None:
            m = {**mk, **km_cached}
        elif c["cond"] in ("caa", "alphasteer") and base_kb is not None and (
                mk["acc_amb"] <= base_kb["acc_amb"] or (base_kb["acc_dis"] - mk["acc_dis"]) * 100 > 3.0):
            # cannot be selected (no ambig gain / disamb drop beyond tolerance): skip the costly KMMLU-val pass
            skip = "no_amb_gain" if mk["acc_amb"] <= base_kb["acc_amb"] else "disamb_drop>3pp"
            m = {**mk, "kmmlu_acc": float("nan"), "kmmlu_rel_drop_pct": float("nan"), "kmmlu_abs_drop_pp": float("nan")}
        else:
            kmp = eval_kmmlu(model, tok, kv, shots, cfg, c["cond"], spec, c["lam"])
            m = summarize(kbp, kmp, km_base)
        if c["cond"] == "base":
            base_kb = mk
        m["kmmlu_skipped"] = skip
        if args.stage == "full":
            kbp.drop(columns=["letter_logits"]).to_parquet(PRED / f"{hashlib.md5(k.encode()).hexdigest()[:12]}.parquet")
        row = {**c, **m, "key": k, "sec": time.time() - t0}
        with open(out_f, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        log.info(f"{c['cond']} {c['r_kind']} L{c['layers']} p{c['p']} a{c['alpha']} lam{c['lam']}: "
                 f"amb={m['acc_amb']:.4f} dis={m['acc_dis']:.4f} bias_amb={m['bias_amb']:.4f} "
                 f"kmmlu={m['kmmlu_acc']:.4f} rel_drop={m['kmmlu_rel_drop_pct']:.2f}% skip={skip} ({row['sec']:.0f}s)")
        if m["kmmlu_rel_drop_pct"] > STOP_REL or (base_kb and (base_kb["acc_dis"] - m["acc_dis"]) * 100 > 5.0):
            log.info("  -> KMMLU rel drop > 3% or disamb drop > 5pp: stop this lambda chain")
            break
