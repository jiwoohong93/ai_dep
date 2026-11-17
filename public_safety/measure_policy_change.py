#!/usr/bin/env python3
"""Measure runtime with and without fairness constraints."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INTERVENTION_SCRIPT = SCRIPT_DIR / "intervention_region.py"


def run_timed(command: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure intervention_region.py runtimes with and without "
            "fairness constraints."
        )
    )
    parser.add_argument(
        "--taus",
        "--tau",
        dest="tau",
        type=float,
        default=0.05,
        help="Fairness constraint value (default: 0.05).",
    )
    parser.add_argument("--n-execution", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if not INTERVENTION_SCRIPT.exists():
        raise FileNotFoundError(
            "Cannot find intervention_region.py next to this script: "
            f"{INTERVENTION_SCRIPT}"
        )

    if args.n_execution < 1:
        parser.error("--n_execution must be at least 1.")

    fair_times = []
    no_fair_times = []
    for _ in range(args.n_execution):
        command = [
            sys.executable,
            str(INTERVENTION_SCRIPT),
            "--taus",
            str(args.tau),
            "--k",
            str(args.k),
            "--quiet",
        ]
        fair_times.append(run_timed(command))

        no_fair_command = [
            sys.executable,
            str(INTERVENTION_SCRIPT),
            "--no-fairness-constraint",
            "--k",
            str(args.k),
            "--quiet",
        ]
        no_fair_times.append(run_timed(no_fair_command))

    average_fair_time = sum(fair_times) / len(fair_times)
    no_fair_time = sum(no_fair_times) / len(no_fair_times)
    increase_pct = (
        (average_fair_time - no_fair_time) / no_fair_time * 100
        if no_fair_time > 0
        else math.nan
    )
    if increase_pct < 0.0:
        increase_pct = 0.0
        print("Increase pct is negative...")

    print("\n\n======================== Comparison Results ========================")
    print("Fairness-constrained runtime measurements")
    for elapsed in fair_times:
        print(f"tau={args.tau:g}: {elapsed:.3f} seconds")
    
    print(
        f"Average across {len(fair_times)} run(s): "
        f"{average_fair_time:.3f} seconds"
    )

    for elapsed in no_fair_times:
        print(f"\nWithout fairness constraint : {elapsed:.3f} seconds")

    print(
        f"Average across {len(no_fair_times)} run(s): "
        f"{no_fair_time:.3f} seconds"
    )

    if math.isfinite(increase_pct):
        print(
            "Average fairness-constrained runtime increase over no-fairness: "
            f"{increase_pct:.2f}%"
        )
    else:
        print("Runtime increase could not be calculated (zero baseline runtime).")


if __name__ == "__main__":
    main()
