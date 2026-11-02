#!/usr/bin/env python3
"""Regional/interference intervention optimization with Gurobi aux variables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def natural_key(name: str, prefix: str) -> int:
    return int(name[len(prefix):])


def sorted_cols(df: pd.DataFrame, pattern: str, prefix: str) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(pattern, c)]
    return sorted(cols, key=lambda c: natural_key(c, prefix))


def coerce_weight_samples(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 3:
        if weights.shape[1] != 1:
            raise ValueError(f"Expected weight shape (samples, 1, d), got {weights.shape}")
        return weights[:, 0, :]
    if weights.ndim == 2:
        return weights
    if weights.ndim == 1:
        return weights.reshape(1, -1)
    raise ValueError(f"Unsupported weight shape: {weights.shape}")


def split_weights(weights: np.ndarray, n_x: int, n_sa: int):
    weights = coerce_weight_samples(weights)
    expected = n_x * n_sa + n_sa + n_sa
    if weights.shape[1] != expected:
        raise ValueError(
            f"Model has {weights.shape[1]} coefficients, but data implies {expected}."
        )
    start_treat = n_x * n_sa
    start_sa = start_treat + n_sa
    beta = weights[:, :start_treat].reshape(weights.shape[0], n_x, n_sa)
    alpha = weights[:, start_treat:start_sa]
    theta = weights[:, start_sa:start_sa + n_sa]
    return beta, alpha, theta


def actual_samples_for_exposure(
    weights: np.ndarray,
    x_i: np.ndarray,
    sensitive_i: np.ndarray,
    exposure: float,
) -> np.ndarray:
    n_x = x_i.shape[0]
    n_sa = sensitive_i.shape[0]
    beta, alpha, theta = split_weights(weights, n_x, n_sa)
    x_part = np.einsum("smk,m,k->s", beta, x_i, sensitive_i)
    treat_part = np.einsum("sk,k->s", alpha, sensitive_i) * exposure
    sa_part = np.einsum("sk,k->s", theta, sensitive_i)
    return x_part + treat_part + sa_part


def counterfactual_samples_for_exposure(
    weights: np.ndarray,
    x_i: np.ndarray,
    exposure: float,
) -> np.ndarray:
    n_x = x_i.shape[0]
    n_sa = infer_n_sa(weights, n_x)
    beta, alpha, theta = split_weights(weights, n_x, n_sa)
    x_part = np.einsum("smk,m->sk", beta, x_i)
    treat_part = alpha * exposure
    return x_part + treat_part + theta


def infer_n_sa(weights: np.ndarray, n_x: int) -> int:
    weights = coerce_weight_samples(weights)
    denom = n_x + 2
    if weights.shape[1] % denom != 0:
        raise ValueError(
            f"Cannot infer sensitive dimension from {weights.shape[1]} coefficients."
        )
    return weights.shape[1] // denom


def parse_taus(args: argparse.Namespace) -> np.ndarray:
    if args.taus:
        return np.array([float(x.strip()) for x in args.taus.split(",") if x.strip()])
    if args.tau is not None:
        return np.array(args.tau, dtype=float)
    if args.tau_file is not None:
        data = np.load(args.tau_file)
        if "TAUS" not in data.files:
            raise ValueError(f"{args.tau_file} does not contain key 'TAUS'")
        return np.asarray(data["TAUS"], dtype=float)
    return np.array([0.04], dtype=float)


def pattern_exposures(neigh_sim_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = neigh_sim_i.shape[0]
    bit_mask = np.zeros((2 ** width, width), dtype=float)
    ints = np.arange(2 ** width, dtype=int)
    for j in range(width):
        bit_mask[:, j] = ints % 2
        ints //= 2
    exposure = np.max(bit_mask * neigh_sim_i[None, :], axis=1)
    return bit_mask, exposure


def compute_pattern_values(
    *,
    w_mean: np.ndarray,
    w_samples: np.ndarray,
    x_i: np.ndarray,
    sensitive_i: np.ndarray,
    exposures: np.ndarray,
    constraint_mode: str,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.empty(exposures.shape[0], dtype=float)
    gaps = np.empty((sensitive_i.shape[0], exposures.shape[0]), dtype=float)

    for f, exposure in enumerate(exposures):
        actual_mean = actual_samples_for_exposure(w_mean, x_i, sensitive_i, exposure)[0]
        cf_mean = counterfactual_samples_for_exposure(w_mean, x_i, exposure)[0]
        weights[f] = actual_mean

        if constraint_mode == "mean":
            gaps[:, f] = actual_mean - cf_mean
        elif constraint_mode == "quantile":
            actual = actual_samples_for_exposure(w_samples, x_i, sensitive_i, exposure)
            cf = counterfactual_samples_for_exposure(w_samples, x_i, exposure)
            gap_samples = actual[:, None] - cf
            gaps[:, f] = np.quantile(gap_samples, 1.0 - q, axis=0)
        else:
            raise ValueError(f"Unknown constraint mode: {constraint_mode}")
    return weights, gaps


def evaluate_solution(
    sol: np.ndarray,
    w_samples: np.ndarray,
    x: np.ndarray,
    sensitive: np.ndarray,
    neigh: np.ndarray,
    neigh_sim: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = coerce_weight_samples(w_samples).shape[0]
    n, n_sa = sensitive.shape
    obj_values = np.zeros(n_samples, dtype=float)
    gap_values = np.zeros((n_samples, n, n_sa), dtype=float)

    z = sol.astype(float)
    for i in range(n):
        exposure = float(np.max(neigh_sim[i] * z[neigh[i]]))
        actual = actual_samples_for_exposure(w_samples, x[i], sensitive[i], exposure)
        cf = counterfactual_samples_for_exposure(w_samples, x[i], exposure)
        obj_values += actual
        gap_values[:, i, :] = actual[:, None] - cf

    max_gap_values = gap_values.max(axis=(1, 2))
    return obj_values, gap_values, max_gap_values


def safe_tau_name(tau: float) -> str:
    return f"{tau:.10g}".replace("-", "m").replace(".", "p")


def quantile_summary_name(q: float) -> str:
    percent = q * 100.0
    if np.isclose(percent, round(percent)):
        return f"obj_val_region_p{int(round(percent)):02d}.csv"
    return f"obj_val_region_p{safe_tau_name(percent)}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Regional fair intervention optimization.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data_new.csv")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "po_bnn_region.npz")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--tau", type=float, action="append", default=None)
    parser.add_argument("--taus", type=str, default=None)
    parser.add_argument("--tau-file", type=Path, default=None)
    parser.add_argument("--constraint-mode", choices=["mean", "quantile"], default="mean")
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-lp", action="store_true")
    args = parser.parse_args()

    if args.constraint_mode == "quantile" and not 0.0 < args.q < 0.5:
        raise ValueError("--q must be between 0 and 0.5 for quantile mode.")

    import gurobipy as gb

    df = pd.read_csv(args.data)
    sa_cols = sorted_cols(df, r"SA\d+", "SA")
    x_cols = sorted_cols(df, r"X\d+", "X")
    neigh_cols = sorted_cols(df, r"neigh\d+", "neigh")
    sim_cols = sorted_cols(df, r"sim\d+", "sim")
    if not (sa_cols and x_cols and neigh_cols and sim_cols):
        raise ValueError("Prepared columns missing. Run data_preparation.py first.")

    sensitive = df[sa_cols].astype(float).to_numpy()
    x = df[x_cols].astype(float).to_numpy()
    neigh = df[neigh_cols].astype(int).to_numpy()
    neigh_sim = df[sim_cols].astype(float).to_numpy()
    model_data = np.load(args.model, allow_pickle=True)
    w_mean = coerce_weight_samples(model_data["w"])
    w_samples = coerce_weight_samples(model_data["w_samples"])
    split_weights(w_mean, x.shape[1], sensitive.shape[1])
    split_weights(w_samples, x.shape[1], sensitive.shape[1])

    taus = parse_taus(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary is not None:
        summary_path = args.summary
    elif args.constraint_mode == "quantile":
        summary_path = args.out_dir / quantile_summary_name(args.q)
    else:
        summary_path = args.out_dir / "obj_val_region.csv"
    rows = []

    for tau in taus:
        tau = float(tau)
        print(f"running tau={tau}")
        model = gb.Model()
        if args.quiet:
            model.Params.OutputFlag = 0

        interventions = model.addVars(np.arange(len(df)), lb=0, ub=1, vtype=gb.GRB.BINARY)
        model.addConstr(gb.quicksum(interventions[i] for i in range(len(df))) <= args.k)

        all_selected_metrics = []
        for index in range(len(df)):
            bit_mask, exposures = pattern_exposures(neigh_sim[index])
            weights, gaps = compute_pattern_values(
                w_mean=w_mean,
                w_samples=w_samples,
                x_i=x[index],
                sensitive_i=sensitive[index],
                exposures=exposures,
                constraint_mode=args.constraint_mode,
                q=args.q,
            )
            all_selected_metrics.append(gaps)
            aux = model.addVars(
                np.arange(bit_mask.shape[0]),
                lb=0,
                ub=1,
                obj=weights,
                vtype=gb.GRB.CONTINUOUS,
            )
            model.update()
            for f in range(bit_mask.shape[0]):
                for j in range(bit_mask.shape[1]):
                    neighbor = int(neigh[index, j])
                    if bit_mask[f, j]:
                        model.addConstr(aux[f] <= interventions[neighbor])
                    else:
                        model.addConstr(aux[f] <= 1 - interventions[neighbor])
            model.addConstr(aux.sum() == 1)
            for r in range(sensitive.shape[1]):
                model.addConstr(
                    gb.quicksum(aux[f] * float(gaps[r, f]) for f in range(gaps.shape[1]))
                    <= tau
                )

        model.setObjective(model.getObjective(), gb.GRB.MAXIMIZE)
        model.optimize()

        tau_name = safe_tau_name(tau)
        stem = f"region_ipm_k{args.k}_tau{tau_name}_{args.constraint_mode}"
        if args.constraint_mode == "quantile":
            stem += f"_q{safe_tau_name(args.q)}"
        if args.write_lp:
            model.write(str(args.out_dir / f"{stem}.lp"))

        if model.status == gb.GRB.Status.OPTIMAL:
            sol = np.array([interventions[i].X for i in range(len(df))]) >= 0.5
            obj_values, gap_values, max_gap_values = evaluate_solution(
                sol, w_samples, x, sensitive, neigh, neigh_sim
            )
            result_path = args.out_dir / f"{stem}.npz"
            np.savez_compressed(
                result_path,
                sol=sol,
                obj_values=obj_values,
                gap_values=gap_values,
                max_gap_values=max_gap_values,
                tau=np.array([tau]),
                q=np.array([args.q]),
                constraint_mode=np.array([args.constraint_mode]),
            )
            row = {
                "tau": tau,
                "status": "OPTIMAL",
                "selected_count": int(sol.sum()),
                "objective_mean_model": float(model.objVal),
                "obj_mean_posterior": float(obj_values.mean()),
                "obj_std_posterior": float(obj_values.std()),
                "max_gap_mean_posterior": float(max_gap_values.mean()),
                "max_gap_std_posterior": float(max_gap_values.std()),
                "result_file": str(result_path),
            }
            selected_sensitive = sensitive[sol].sum(axis=0)
            for col, value in zip(sa_cols, selected_sensitive):
                row[f"selected_{col}"] = float(value)
            rows.append(row)
            print(f"  OPTIMAL: selected={int(sol.sum())}, objective={model.objVal:.6f}")
        else:
            rows.append(
                {
                    "tau": tau,
                    "status": f"GUROBI_STATUS_{model.status}",
                    "selected_count": 0,
                    "objective_mean_model": np.nan,
                    "obj_mean_posterior": np.nan,
                    "obj_std_posterior": np.nan,
                    "max_gap_mean_posterior": np.nan,
                    "max_gap_std_posterior": np.nan,
                    "result_file": "",
                }
            )
            print(f"  did not solve optimally, status={model.status}")

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
