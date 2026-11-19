import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import compare_reference, load_model, run_evaluation
from run_experiment import (
    VectorizedLoader,
    build_samples,
    clean_value,
    fairness_metrics,
    make_torch_components,
    normalize_header,
    parse_args,
)

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_cleaning(self):
        self.assertEqual(normalize_header(" Job ID \n"), "JobID")
        self.assertIsNone(clean_value(float("nan")))
        self.assertIsNone(clean_value(" null "))
        self.assertEqual(clean_value(" sales\n  planning "), "sales planning")

    def test_final_config_and_cli_override(self):
        args = parse_args(["--config", str(ROOT / "configs/experiment.json"), "--epochs", "2"])
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.patience, 4)
        self.assertEqual(args.preserve_rank, 128)
        self.assertEqual(args.edit_lr, 0.01)
        self.assertFalse(args.use_data_parallel)
        self.assertEqual(args.out_dir, "artifacts")
        self.assertEqual(len(args.embedding_revision), 40)

    def test_unknown_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"preserve_rnak": 128}), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config", str(path)])

    def test_invalid_batch_size_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--batch-size", "0"])

    def test_samples_are_balanced_reproducible_and_gender_is_per_applicant(self):
        applicants = pd.DataFrame({"resume_id": range(20), "applicant_id": [i // 2 for i in range(20)]})
        jobs = pd.DataFrame({"job_id": range(40), "company_id": range(40)})
        applications = pd.DataFrame({
            "resume_id": range(20), "applicant_id": [i // 2 for i in range(20)],
            "job_id": range(20), "company_id": range(20),
        })
        first = build_samples(applications, applicants, jobs, neg_ratio=1, seed=42)
        second = build_samples(applications, applicants, jobs, neg_ratio=1, seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["label"].value_counts().to_dict(), {1: 20, 0: 20})
        self.assertEqual(first["split"].value_counts().to_dict(), {"train": 32, "val": 4, "test": 4})
        self.assertTrue((first.groupby("applicant_id")["gender"].nunique() == 1).all())
        self.assertTrue((first.loc[first["label"] == 0, "resume_id"] != first.loc[first["label"] == 0, "job_id"]).all())

    def test_fairness_aligns_indices_and_excludes_single_company_resumes(self):
        samples = pd.DataFrame({"resume_id": ["r1", "r1", "r2", "r3", "r3"],
                                "company_id": ["a", "b", "a", "a", "a"]})
        original = {"sample_indices": np.arange(5), "probs": np.array([.9, .1, .3, .4, .5]),
                    "preds": np.array([1, 0, 0, 0, 1])}
        order = np.array([4, 2, 0, 3, 1])
        flipped = {"sample_indices": np.arange(5)[order],
                   "probs": np.array([.1, .9, .9, .6, .1])[order],
                   "preds": np.array([0, 1, 1, 1, 0])[order]}
        metrics = fairness_metrics(samples, original, flipped)
        self.assertEqual(metrics["top1_company_eval_resumes"], 3)
        self.assertEqual(metrics["top1_company_multi_company_eval_resumes"], 1)
        self.assertEqual(metrics["top1_company_consistency_accuracy_multi_company_resumes"], 0)
        self.assertAlmostEqual(metrics["top1_company_consistency_accuracy_all_resumes"], 2 / 3)
        flipped["sample_indices"] = np.arange(1, 6)
        with self.assertRaises(ValueError):
            fairness_metrics(samples, original, flipped)

    def test_vectorized_loader_preserves_row_indices(self):
        embeddings = np.arange(24, dtype=np.float16).reshape(3, 8)
        loader = VectorizedLoader("test", np.array([2, 0]), np.array([0, 1, 2]),
                                  np.array([2, 1, 0]), np.array([1, 0, 1]),
                                  np.array([0, 1, 0]), embeddings, embeddings, 1, False, 42)
        batches = list(loader)
        self.assertEqual([batch[0].item() for batch in batches], [2, 0])
        np.testing.assert_array_equal(batches[0][1].numpy(), embeddings[[2]])
        np.testing.assert_array_equal(batches[0][2].numpy(), embeddings[[0]])

    def test_private_checkpoint_evaluation_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"resume_id": ["r0", "r1"]}).to_parquet(path / "applicants.parquet", index=False)
            pd.DataFrame({"job_id": ["j0", "j1"], "company_id": ["c0", "c1"]}).to_parquet(path / "jobs.parquet", index=False)
            samples = pd.DataFrame({"resume_id": ["r0"] * 6, "job_id": ["j0", "j1"] * 3,
                                    "company_id": ["c0", "c1"] * 3, "label": [1, 0] * 3,
                                    "gender": [0] * 6, "split": ["train", "train", "val", "val", "test", "test"]})
            samples.to_parquet(path / "samples.parquet", index=False)
            rng = np.random.default_rng(42)
            for name in ("applicants", "jobs"):
                np.save(path / f"{name}_embeddings.npy", rng.normal(size=(2, 8)).astype(np.float16))
            settings = {"embedding_model": "synthetic", "neg_ratio": 1, "seed": 42,
                        "hidden_dim": 16, "preserve_rank": 4, "epochs": 1, "dropout": 0}
            model = make_torch_components()(8, hidden_dim=16, dropout=0)
            checkpoint = {"state_dict": model.state_dict(), "args": settings}
            for name in ("base_model.pt", "alphaedit_fair_model.pt"):
                torch.save(checkpoint, path / name)
            loaded, _ = load_model(path / "base_model.pt", 8, "cpu")
            self.assertFalse(loaded.training)
            args = SimpleNamespace(artifact_dir=path, batch_size=2, device="cpu",
                                   base_checkpoint=None, edited_checkpoint=None)
            report = run_evaluation(args)
            self.assertEqual(report["counts"]["test"], 2)
            self.assertEqual(report["base"]["fairness"]["top1_company_multi_company_eval_resumes"], 1)
            compare_reference(report, copy.deepcopy(report))
            changed = copy.deepcopy(report)
            changed["base"]["recruitment_accuracy"] = float("nan")
            with self.assertRaises(ValueError):
                compare_reference(changed, report)
            changed = copy.deepcopy(report)
            changed["counts"]["test"] += 1
            with self.assertRaises(ValueError):
                compare_reference(changed, report)
            samples["split"] = "train"
            samples.to_parquet(path / "samples.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "empty"):
                run_evaluation(args)


if __name__ == "__main__":
    unittest.main()
