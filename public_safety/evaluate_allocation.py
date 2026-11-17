#!/usr/bin/env python3
"""Run fair and no-fairness intervention optimizations and compare their PO."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INTERVENTION_SCRIPT = SCRIPT_DIR / "intervention_region.py"
RESULTS_DIR = SCRIPT_DIR / "results"


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def read_csv_row(path: Path, tau: float | None = None) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Result CSV was not created: {path}")

    df = pd.read_csv(path)
    required = {"tau", "status", "objective_mean_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if tau is None:
        selected = df[df["tau"].isna()]
        if selected.empty:
            # Some CSV writers or prior versions may encode an empty tau
            # differently. The no-fairness summary should contain one row.
            if len(df) == 1:
                selected = df
            else:
                raise ValueError(f"Could not find the no-fairness row in {path}")
    else:
        tau_values = pd.to_numeric(df["tau"], errors="coerce")
        selected = df[tau_values.map(lambda value: math.isclose(
            value, tau, rel_tol=1e-9, abs_tol=1e-12
        ))]
        if selected.empty:
            raise ValueError(f"Could not find tau={tau} in {path}")

    if len(selected) > 1:
        raise ValueError(f"More than one matching row found in {path}")
    return selected.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run intervention_region.py with and without fairness constraints, "
            "then compare objective_mean_model."
        )
    )
    parser.add_argument(
        "--taus",
        required=True,
        help="Comma-separated tau value(s), e.g. --taus 0.05 or --taus 0.03,0.05",
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if not INTERVENTION_SCRIPT.exists():
        raise FileNotFoundError(
            f"Cannot find intervention_region.py next to this script: "
            f"{INTERVENTION_SCRIPT}"
        )

    taus = [float(value.strip()) for value in args.taus.split(",") if value.strip()]
    if not taus:
        raise ValueError("Provide at least one tau value with --taus.")

    # First create the constrained result CSV, then the no-fairness CSV.
    run_command([
        sys.executable,
        str(INTERVENTION_SCRIPT),
        "--taus",
        ",".join(str(tau) for tau in taus),
        "--k",
        str(args.k),
        "--quiet",
    ])
    run_command([
        sys.executable,
        str(INTERVENTION_SCRIPT),
        "--no-fairness-constraint",
        "--k",
        str(args.k),
        "--quiet",
    ])

    print("\n\n======================== Comparison Results ========================")
    fair_csv = RESULTS_DIR / "obj_val_region.csv"
    no_fair_csv = RESULTS_DIR / "obj_val_region_no_fairness.csv"
    no_fair_row = read_csv_row(no_fair_csv)

    if no_fair_row["status"] == "OPTIMAL":
        print("No Fairness constraint : OPTIMAL solution found")
    else:
        print(
            "No Fairness constraint : "
            f"status={no_fair_row['status']} (not OPTIMAL)"
        )

    fair_rows = [read_csv_row(fair_csv, tau) for tau in taus]
    for tau, fair_row in zip(taus, fair_rows):
        if fair_row["status"] == "OPTIMAL":
            print(f"taus {tau:g} : OPTIMAL solution found")
        else:
            print(f"taus {tau:g} : status={fair_row['status']} (not OPTIMAL)")

    if no_fair_row["status"] != "OPTIMAL":
        print("Cannot calculate the decrease: no-fairness solution is not OPTIMAL.")
        return

    no_fair_objective = float(no_fair_row["objective_mean_model"])
    if not math.isfinite(no_fair_objective) or no_fair_objective == 0:
        print("Cannot calculate the percentage decrease: invalid baseline objective.")
        return

    for tau, fair_row in zip(taus, fair_rows):
        if fair_row["status"] != "OPTIMAL":
            print(f"tau={tau:g}: skipped comparison because status is not OPTIMAL.")
            continue

        fair_objective = float(fair_row["objective_mean_model"])
        decrease_pct = (no_fair_objective - fair_objective) / no_fair_objective * 100
        print(
            f"tau={tau:g}: objective_mean_model={fair_objective:.10f}; "
            f"without fairness={no_fair_objective:.10f}; "
            f"decrease={decrease_pct:.4f}%"
        )


if __name__ == "__main__":
    main()
