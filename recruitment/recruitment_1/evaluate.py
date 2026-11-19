#!/usr/bin/env python
"""Evaluate private checkpoints without training or exporting individual records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from run_experiment import (
    VectorizedLoader,
    evaluate_model,
    fairness_metrics,
    load_training_arrays,
    make_torch_components,
)


def load_model(path: Path, embedding_dim: int, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    settings = checkpoint.get("args", {})
    model = make_torch_components()(
        emb_dim=embedding_dim,
        proj_dim=state["resume_proj.weight"].shape[0],
        hidden_dim=state["head.weight"].shape[1],
        dropout=settings.get("dropout", 0.15),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), settings


def compare_reference(report: dict, reference: dict, tolerance: float = 1e-6) -> None:
    if report["counts"] != reference["counts"]:
        raise ValueError("Evaluation counts differ from the reference split.")
    for name in ("base", "alphaedit_nullspace"):
        observed = report[name]
        expected = reference[name]
        comparisons = [("recruitment_accuracy", observed["recruitment_accuracy"], expected["recruitment_accuracy"])]
        comparisons.extend(
            (key, observed["fairness"].get(key), value)
            for key, value in expected["fairness"].items()
        )
        for key, actual, value in comparisons:
            if value is None or actual is None:
                matches = value == actual
            else:
                matches = math.isfinite(actual) and abs(actual - value) <= tolerance
            if not matches:
                raise ValueError(f"Reference mismatch: {name}.{key}")


def run_evaluation(args) -> dict:
    artifact_dir = Path(args.artifact_dir)
    original_count = len(pd.read_parquet(artifact_dir / "samples.parquet", columns=["label"]))
    samples, resumes, jobs = load_training_arrays(SimpleNamespace(out_dir=artifact_dir))
    if len(samples) != original_count:
        raise ValueError("Some samples are missing embeddings; evaluation cannot drop records.")
    if resumes.ndim != 2 or jobs.ndim != 2 or resumes.shape[1] != jobs.shape[1]:
        raise ValueError("Resume and job embedding dimensions must match.")
    if not np.isin(samples["split"], ["train", "val", "test"]).all():
        raise ValueError("Unknown split values in samples.parquet.")
    if not samples["gender"].isin([0, 1]).all() or not samples["label"].isin([0, 1]).all():
        raise ValueError("Labels and gender evaluation labels must be binary.")
    test_indices = np.flatnonzero(samples["split"].to_numpy() == "test")
    if len(test_indices) == 0:
        raise ValueError("The test split is empty.")
    loader = VectorizedLoader(
        name="test",
        sample_indices=test_indices,
        resume_indices=samples["resume_idx"].to_numpy(np.int64),
        job_indices=samples["job_idx"].to_numpy(np.int64),
        labels=samples["label"].to_numpy(np.float32),
        genders=samples["gender"].to_numpy(np.float32),
        applicant_embeddings=resumes,
        job_embeddings=jobs,
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
    )
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    torch.set_float32_matmul_precision("high")
    report = {
        "counts": {
            "samples": int(len(samples)),
            "train": int((samples["split"] == "train").sum()),
            "val": int((samples["split"] == "val").sum()),
            "test": int(len(test_indices)),
            "test_positive": int((samples.iloc[test_indices]["label"] == 1).sum()),
            "test_negative": int((samples.iloc[test_indices]["label"] == 0).sum()),
        }
    }
    paths = {
        "base": args.base_checkpoint or artifact_dir / "base_model.pt",
        "alphaedit_nullspace": args.edited_checkpoint or artifact_dir / "alphaedit_fair_model.pt",
    }
    for name, path in paths.items():
        model, settings = load_model(Path(path), resumes.shape[1], device)
        original = evaluate_model(model, loader, device)
        flipped = evaluate_model(model, loader, device, flip_gender=True)
        report[name] = {
            "recruitment_accuracy": original["accuracy"],
            "fairness": fairness_metrics(samples, original, flipped),
        }
        if name == "alphaedit_nullspace":
            keys = ("embedding_model", "neg_ratio", "seed", "hidden_dim", "preserve_rank")
            report["settings"] = {key: settings[key] for key in keys}
            report["settings"].update(
                split="pair-level stratified 80/10/10 with random_state seed",
                batch_size=args.batch_size,
                epochs_requested=settings["epochs"],
            )
        del model
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--edited-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--check-targets", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.tolerance < 0:
        parser.error("Batch size must be positive and tolerance must be nonnegative.")
    report = run_evaluation(args)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if args.reference:
        compare_reference(report, json.loads(args.reference.read_text(encoding="utf-8")), args.tolerance)
        print("Reference metrics: PASS")
    if args.check_targets:
        edited = report["alphaedit_nullspace"]
        accuracy = edited["recruitment_accuracy"]
        fairness = edited["fairness"]["top1_company_consistency_accuracy_multi_company_resumes"]
        passed = accuracy >= 0.80 and fairness is not None and fairness >= 0.95
        print("Phase-2 targets: " + ("PASS" if passed else "FAIL"))
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
