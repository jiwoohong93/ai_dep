#!/usr/bin/env python3
"""Prepare regional/interference data for the IPM + Bayesian PO pipeline.

Expected input columns:
    SA01, SA02, ... : sensitive attribute fractions or percentages
    X01, X02, ...   : non-treatment covariates
    treat           : binary observed treatment
    objective       : observed outcome
    long, lat       : coordinates

Output:
    data_new.csv with neighbor indices, neighbor similarities, own covariates,
    and sensitive-attribute interaction features.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def natural_key(name: str, prefix: str) -> int:
    return int(name[len(prefix):])


def find_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    sa_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"SA\d+", c)],
        key=lambda c: natural_key(c, "SA"),
    )
    x_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"X\d+", c)],
        key=lambda c: natural_key(c, "X"),
    )
    missing = [c for c in ("treat", "objective", "long", "lat") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    if not sa_cols:
        raise ValueError("No sensitive attribute columns found. Expected SA01, SA02, ...")
    if not x_cols:
        raise ValueError("No X columns found. Expected X01, X02, ...")
    return sa_cols, x_cols


def parse_treatment(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        treat = series.astype(float).to_numpy()
    else:
        mapping = {
            "1": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "n": 0.0,
        }
        treat = series.astype(str).str.strip().str.lower().map(mapping).to_numpy()
    if np.isnan(treat).any():
        raise ValueError("treat must be binary: 0/1 or Yes/No style values.")
    unique = set(np.unique(treat))
    if not unique <= {0.0, 1.0}:
        raise ValueError(f"treat must be binary, got values {sorted(unique)}")
    return treat.astype(float)


def scale_sensitive(values: np.ndarray) -> np.ndarray:
    sensitive = values.astype(float).copy()
    if np.nanmax(sensitive) > 1.0 + 1e-8:
        sensitive = sensitive / 100.0
    sensitive[np.isclose(sensitive, 0.0, atol=1e-12)] = 0.0
    if np.any(sensitive < 0):
        raise ValueError("Sensitive attribute columns must be nonnegative.")
    row_sum = sensitive.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Every row must have positive sensitive attribute mass.")
    return sensitive


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_neighbors(lat: np.ndarray, lon: np.ndarray, sim: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(lat)
    dist = haversine_km(lat, lon)
    similarity = 1.0 / (1.0 + dist)
    np.fill_diagonal(similarity, 1.0)

    neigh = np.zeros((n, sim + 1), dtype=int)
    neigh_sim = np.zeros((n, sim + 1), dtype=float)
    for i in range(n):
        order = np.argsort(-similarity[i])
        chosen = order[: sim + 1]
        neigh[i] = chosen
        neigh_sim[i] = similarity[i, chosen]
    return neigh, neigh_sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare regional data with neighbor indices.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data.csv")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "data_new.csv")
    parser.add_argument("--meta-out", type=Path, default=SCRIPT_DIR / "data_new_meta.json")
    parser.add_argument("--sim", type=int, default=5, help="Number of nearest neighbors excluding self.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    sa_cols, x_cols = find_columns(df)
    sensitive = scale_sensitive(df[sa_cols].to_numpy())
    x = df[x_cols].astype(float).to_numpy()
    treat = parse_treatment(df["treat"])
    objective = df["objective"].astype(float).to_numpy()
    lat = df["lat"].astype(float).to_numpy()
    lon = df["long"].astype(float).to_numpy()

    neigh, neigh_sim = build_neighbors(lat, lon, args.sim)

    out = pd.DataFrame({"unit_id": np.arange(len(df)), "objective": objective, "treat": treat})
    out["long"] = lon
    out["lat"] = lat

    for idx, col in enumerate(sa_cols):
        out[col] = sensitive[:, idx]

    for idx, col in enumerate(x_cols):
        out[col] = x[:, idx]
    out["TEXP"] = treat

    xf_cols = []
    for x_idx, x_col in enumerate(x_cols):
        for sa_idx, sa_col in enumerate(sa_cols):
            name = f"XF_{x_col}_{sa_col}"
            xf_cols.append(name)
            out[name] = x[:, x_idx] * sensitive[:, sa_idx]

    tf_cols = []
    for sa_idx, sa_col in enumerate(sa_cols):
        name = f"TF_{sa_col}"
        tf_cols.append(name)
        out[name] = treat * sensitive[:, sa_idx]

    neigh_cols = []
    sim_cols = []
    for j in range(args.sim + 1):
        neigh_col = f"neigh{j}"
        sim_col = f"sim{j}"
        neigh_cols.append(neigh_col)
        sim_cols.append(sim_col)
        out[neigh_col] = neigh[:, j]
        out[sim_col] = neigh_sim[:, j]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    meta = {
        "source": str(args.data),
        "n": int(len(df)),
        "sim": int(args.sim),
        "sensitive_columns": sa_cols,
        "x_columns": x_cols,
        "xf_columns": xf_cols,
        "tf_columns": tf_cols,
        "neighbor_columns": neigh_cols,
        "similarity_columns": sim_cols,
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved prepared data: {args.out}")
    print(f"n={len(df)}, sim={args.sim}, SA={len(sa_cols)}, X={len(x_cols)}")
    print(f"treat count={int(treat.sum())}, control count={int((1 - treat).sum())}")


if __name__ == "__main__":
    main()
