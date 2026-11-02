#!/usr/bin/env python3
"""Estimate IPM/SIPM balancing weights from prepared regional data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent


def natural_key(name: str, prefix: str) -> int:
    return int(name[len(prefix):])


def find_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    sa_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"SA\d+", c)],
        key=lambda c: natural_key(c, "SA"),
    )
    xf_cols = [c for c in df.columns if c.startswith("XF_")]
    if "treat" not in df.columns:
        raise ValueError("Missing required column: treat")
    if not sa_cols:
        raise ValueError("No SA columns found.")
    if not xf_cols:
        raise ValueError("No XF_ columns found. Run data_preparation.py first.")
    return sa_cols, xf_cols


class ParaSigmoid(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x).squeeze())


class SoftmaxWeights(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.vec = nn.Parameter(torch.ones(n))
        self.softmax = nn.Softmax(dim=0)

    def weight(self) -> torch.Tensor:
        return self.softmax(self.vec)


def ipm_loss(
    x_control: torch.Tensor,
    x_treated: torch.Tensor,
    critic: ParaSigmoid,
    w_control: SoftmaxWeights,
    w_treated: SoftmaxWeights,
) -> torch.Tensor:
    n_c, n_t = x_control.shape[0], x_treated.shape[0]
    sig_c = critic(x_control)
    sig_t = critic(x_treated)
    pooled_mean = (torch.sum(sig_c) + torch.sum(sig_t)) / (n_c + n_t)
    control_mean = torch.sum(sig_c * w_control.weight())
    treated_mean = torch.sum(sig_t * w_treated.weight())
    return (pooled_mean - control_mean) ** 2 + (pooled_mean - treated_mean) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate regional IPM balancing weights.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data_new.csv")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "w_IPM_region.npy")
    parser.add_argument("--meta-out", type=Path, default=SCRIPT_DIR / "w_IPM_region_meta.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-critics", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--weight-steps", type=int, default=1)
    parser.add_argument("--critic-lr", type=float, default=1.0)
    parser.add_argument("--weight-lr", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    df = pd.read_csv(args.data)
    sa_cols, xf_cols = find_columns(df)
    cov = df[xf_cols + sa_cols].astype(float).to_numpy()
    treat = df["treat"].astype(float).to_numpy()
    unique = set(np.unique(treat))
    if not unique <= {0.0, 1.0}:
        raise ValueError(f"treat must be binary, got values {sorted(unique)}")

    mean = cov.mean(axis=0, keepdims=True)
    std = cov.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    cov_std = (cov - mean) / std

    treated_idx = np.where(treat == 1.0)[0]
    control_idx = np.where(treat == 0.0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Both treated and control units are required.")

    x_control = torch.tensor(cov_std[control_idx], dtype=torch.float32, device=device)
    x_treated = torch.tensor(cov_std[treated_idx], dtype=torch.float32, device=device)
    d = cov_std.shape[1]

    w_control = SoftmaxWeights(len(control_idx)).to(device)
    w_treated = SoftmaxWeights(len(treated_idx)).to(device)
    critics = [ParaSigmoid(d).to(device) for _ in range(args.n_critics)]
    critic_optimizers = [
        optim.SGD(c.parameters(), lr=args.critic_lr, momentum=0.9, weight_decay=5e-4)
        for c in critics
    ]
    control_optimizer = optim.Adam(w_control.parameters(), lr=args.weight_lr)
    treated_optimizer = optim.Adam(w_treated.parameters(), lr=args.weight_lr)

    last_loss = None
    for _ in range(args.epochs):
        for _ in range(args.critic_steps):
            for critic, optimizer in zip(critics, critic_optimizers):
                loss = -ipm_loss(x_control, x_treated, critic, w_control, w_treated)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        for _ in range(args.weight_steps):
            losses = [
                ipm_loss(x_control, x_treated, critic, w_control, w_treated)
                for critic in critics
            ]
            loss = torch.stack(losses).max()
            control_optimizer.zero_grad()
            treated_optimizer.zero_grad()
            loss.backward()
            control_optimizer.step()
            treated_optimizer.step()
            last_loss = float(loss.detach().cpu())

    weights = np.ones(len(treat), dtype=float)
    weights[control_idx] = w_control.weight().detach().cpu().numpy()
    weights[treated_idx] = w_treated.weight().detach().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, weights)

    meta = {
        "data": str(args.data),
        "sensitive_columns": sa_cols,
        "covariate_columns": xf_cols + sa_cols,
        "n": int(len(treat)),
        "n_treated": int(len(treated_idx)),
        "n_control": int(len(control_idx)),
        "weight_sum": float(weights.sum()),
        "last_max_ipm_loss": last_loss,
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved weights: {args.out}")
    print(f"n={len(treat)}, treated={len(treated_idx)}, control={len(control_idx)}")
    print(f"weight sum={weights.sum():.6f}, min={weights.min():.6g}, max={weights.max():.6g}")


if __name__ == "__main__":
    main()
