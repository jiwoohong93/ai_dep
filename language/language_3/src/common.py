"""Config, seeding, logging, model loading shared by all scripts."""
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path=None, overrides=None):
    path = path or os.environ.get("KOBBQ_CONFIG") or str(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    cfg = _load_yaml(path)
    if overrides:
        cfg = _merge(cfg, overrides)
    # relative paths are relative to the repository root, not the working directory
    repo = Path(__file__).resolve().parents[1]
    cfg["paths"] = {k: str(repo / v) for k, v in cfg["paths"].items()}
    # set only after merging, so an override file's hf_home wins over its base file's
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_home"])
    return cfg


def _load_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:  # model-specific override files point at default.yaml
        cfg = _merge(_load_yaml(str(Path(path).parent / base)), cfg)
    return cfg


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name, logfile=None):
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def run_dir(cfg, kind):
    """Per-model directory for large artifacts on NAS (kind in {acts, steer}) or results."""
    d = Path(cfg["paths"][kind]) / cfg["run_name"] if kind != "results" else Path(cfg["paths"]["results"]) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_model(cfg, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=getattr(torch, cfg["model"]["dtype"]), device_map=device,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"))
    model.eval()
    return model, tok


def decoder_layers(model):
    return model.model.layers


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)
