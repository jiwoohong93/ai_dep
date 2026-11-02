#!/usr/bin/env python3
"""Estimate a Bayesian PO model from prepared regional/interference data."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim


SCRIPT_DIR = Path(__file__).resolve().parent


def natural_key(name: str, prefix: str) -> int:
    return int(name[len(prefix):])


def find_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    sa_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"SA\d+", c)],
        key=lambda c: natural_key(c, "SA"),
    )
    x_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"X\d+", c)],
        key=lambda c: natural_key(c, "X"),
    )
    xf_cols = [c for c in df.columns if c.startswith("XF_")]
    tf_cols = [c for c in df.columns if c.startswith("TF_")]
    if "objective" not in df.columns:
        raise ValueError("Missing required column: objective")
    if not (sa_cols and x_cols and xf_cols and tf_cols):
        raise ValueError("Prepared columns missing. Run data_preparation.py first.")
    return sa_cols, x_cols, xf_cols, tf_cols


class LinearPO(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class SGLD(optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])
                noise = torch.randn_like(p.data) * math.sqrt(2.0 * group["lr"])
                p.data.add_(grad, alpha=-group["lr"])
                p.data.add_(noise)
        return loss


def train_sgld(
    model: LinearPO,
    x: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    lr: float,
    burn_in: int,
    n_samples: int,
    thin: int,
    weight_decay: float,
) -> np.ndarray:
    random.seed(0)
    torch.manual_seed(0)
    optimizer = SGLD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss(reduction="none")
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(1, total_steps + 1):
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.sum(loss_func(pred, y).squeeze() * sample_weight)
        loss.backward()
        optimizer.step()

        if step > burn_in and (step - burn_in) % thin == 0:
            params = list(model.parameters())[0]
            samples.append(params.detach().cpu().numpy().copy())
            print(f"Sample {len(samples)}/{n_samples}, loss: {loss.item():.6f}")
        elif step <= burn_in and step % 200 == 0:
            print(f"Burn-in {step}/{burn_in}, loss: {loss.item():.6f}")

    return np.array(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate regional Bayesian PO coefficients.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data_new.csv")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "w_IPM_region.npy")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "po_bnn_region.npz")
    parser.add_argument("--meta-out", type=Path, default=SCRIPT_DIR / "po_bnn_region_meta.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adam-steps", type=int, default=500)
    parser.add_argument("--adam-lr", type=float, default=1e-2)
    parser.add_argument("--sgld-lr", type=float, default=1e-8)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--thin", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    df = pd.read_csv(args.data)
    sa_cols, x_cols, xf_cols, tf_cols = find_columns(df)
    feature_cols = xf_cols + tf_cols + sa_cols
    features = df[feature_cols].astype(float).to_numpy()
    objective = df["objective"].astype(float).to_numpy().reshape(-1, 1)

    ipm_weights = np.load(args.weights)
    if ipm_weights.shape[0] != features.shape[0]:
        raise ValueError(
            f"Weight length {ipm_weights.shape[0]} does not match rows {features.shape[0]}"
        )
    sample_weight = np.sqrt(ipm_weights.astype(float))
    sample_weight = sample_weight / sample_weight.sum()
    # sample_weight = ipm_weights.astype(float)

    x_t = torch.tensor(features, dtype=torch.float32, device=device)
    y_t = torch.tensor(objective, dtype=torch.float32, device=device)
    w_t = torch.tensor(sample_weight, dtype=torch.float32, device=device)

    model = LinearPO(features.shape[1]).to(device)
    loss_func = nn.MSELoss(reduction="none")

    print("Phase 1: Adam pre-training...")
    adam = optim.Adam(model.parameters(), lr=args.adam_lr)
    for step in range(1, args.adam_steps + 1):
        adam.zero_grad()
        pred = model(x_t)
        loss = torch.sum(loss_func(pred, y_t).squeeze() * w_t)
        loss.backward()
        adam.step()
        if step % 100 == 0:
            print(f"  Adam step {step}, loss: {loss.item():.6f}")

    print("Phase 2: SGLD sampling...")
    w_samples = train_sgld(
        model,
        x_t,
        y_t,
        w_t,
        lr=args.sgld_lr,
        burn_in=args.burn_in,
        n_samples=args.n_samples,
        thin=args.thin,
        weight_decay=args.weight_decay,
    )
    w_mean = w_samples.mean(axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        w=w_mean,
        w_samples=w_samples,
        feature_columns=np.array(feature_cols),
        sensitive_columns=np.array(sa_cols),
        x_columns=np.array(x_cols),
        xf_columns=np.array(xf_cols),
        tf_columns=np.array(tf_cols),
        n_sensitive=np.array([len(sa_cols)]),
        n_x=np.array([len(x_cols)]),
        feature_order=np.array(["XF", "TF", "SA"]),
    )

    meta = {
        "data": str(args.data),
        "weights": str(args.weights),
        "sensitive_columns": sa_cols,
        "x_columns": x_cols,
        "feature_columns": feature_cols,
        "n": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "w_mean_shape": list(w_mean.shape),
        "w_samples_shape": list(w_samples.shape),
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved posterior model: {args.out}")
    print(f"w_mean shape: {w_mean.shape}")
    print(f"w_samples shape: {w_samples.shape}")


if __name__ == "__main__":
    main()
