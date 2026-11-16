"""Small independent correctness checks; no downloaded model or GPU needed."""

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import InferenceTagger, load_checkpoint, sha256, write_result
from legacy_decode import legal_bio_logits
from test import evaluate, validate_manifest


class DecoderTests(unittest.TestCase):
    labels = ["O", "B-PS", "I-PS", "B-OG", "I-OG"]

    def legal(self, path):
        previous = "O"
        for index in path:
            tag = self.labels[index]
            if tag.startswith("I-") and previous not in ("B-" + tag[2:], tag):
                return False
            previous = tag
        return True

    def test_viterbi_matches_exhaustive_search(self):
        for seed in range(12):
            generator = torch.Generator().manual_seed(seed)
            logits = torch.randn(2, 7, 5, generator=generator)
            mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]])
            actual = legal_bio_logits(logits, mask, self.labels)
            for row, n in enumerate([3, 2]):
                paths = [
                    p for p in itertools.product(range(5), repeat=n) if self.legal(p)
                ]
                best = max(
                    sum(float(logits[row, j + 1, c]) for j, c in enumerate(p))
                    for p in paths
                )
                chosen = actual[row, 1 : n + 1].argmax(-1).tolist()
                self.assertTrue(self.legal(chosen))
                self.assertAlmostEqual(
                    sum(float(logits[row, j + 1, c]) for j, c in enumerate(chosen)),
                    best,
                    places=5,
                )
                self.assertTrue(torch.equal(actual[row, 0], logits[row, 0]))
                self.assertTrue(torch.equal(actual[row, n + 1 :], logits[row, n + 1 :]))

    def test_starts_and_type_switches(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 10.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 9.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        path = (
            legal_bio_logits(logits, torch.ones(1, 4), self.labels)[0, 1:3]
            .argmax(-1)
            .tolist()
        )
        self.assertTrue(self.legal(path))

    def test_label_reordering_and_log_softmax_invariance(self):
        torch.manual_seed(901)
        logits = torch.randn(1, 6, 5)
        mask = torch.ones(1, 6)
        first = legal_bio_logits(logits, mask, self.labels).argmax(-1)[0, 1:-1]
        permutation = [4, 2, 0, 3, 1]
        second = legal_bio_logits(
            logits[:, :, permutation], mask, [self.labels[i] for i in permutation]
        ).argmax(-1)[0, 1:-1]
        self.assertEqual(first.tolist(), [permutation[i] for i in second.tolist()])
        third = legal_bio_logits(logits.log_softmax(-1), mask, self.labels).argmax(-1)[
            0, 1:-1
        ]
        self.assertTrue(torch.equal(first, third))

    def test_empty_input_and_deterministic_tie(self):
        logits = torch.zeros(1, 5, 5)
        self.assertTrue(
            torch.equal(
                logits,
                legal_bio_logits(logits, torch.tensor([[1, 1, 0, 0, 0]]), self.labels),
            )
        )
        path = legal_bio_logits(logits, torch.ones(1, 5), self.labels).argmax(-1)
        self.assertEqual(path[0, 1:-1].tolist(), [0, 0, 0])

    def test_bad_masks_shapes_labels_and_scores_are_rejected(self):
        logits = torch.zeros(1, 4, 5)
        for mask in [[[1, 0, 1, 0]], [[0, 1, 1, 0]], [[1, 2, 0, 0]], [[1, 0, 0, 0]]]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                legal_bio_logits(logits, torch.tensor(mask), self.labels)
        for labels in [
            ["O", "O", "B-PS", "I-PS", "B-OG"],
            ["O", "I-XX", "B-PS", "I-PS", "B-OG"],
        ]:
            with self.assertRaises(ValueError):
                legal_bio_logits(logits, torch.ones(1, 4), labels)
        with self.assertRaises(ValueError):
            legal_bio_logits(logits, torch.ones(1, 3), self.labels)
        with self.assertRaises(ValueError):
            legal_bio_logits(logits.long(), torch.ones(1, 4), self.labels)
        for value in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                legal_bio_logits(logits + value, torch.ones(1, 4), self.labels)


class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        # Intentionally has no labels argument: forwarding gold would fail.
        logits = torch.stack(
            [input_ids.float() % 2, input_ids.float() % 3, input_ids.float() % 5],
            dim=-1,
        )
        return SimpleNamespace(logits=logits + self.anchor)


class InferenceTests(unittest.TestCase):
    def test_gold_labels_never_reach_model_or_decoder(self):
        x = torch.tensor([[2, 11, 12, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        for decoder in ["argmax", "legal_bio"]:
            model = InferenceTagger(
                DummyClassifier(), ["O", "B-PS", "I-PS"], decoder, 2, 3
            )
            a = model(x, mask, labels=torch.zeros_like(x))[1]
            b = model(x, mask, labels=torch.full_like(x, -100))[1]
            self.assertTrue(torch.equal(a, b))

    def test_bad_layout_and_unknown_decoder_fail(self):
        with self.assertRaises(ValueError):
            InferenceTagger(DummyClassifier(), ["O", "B-PS", "I-PS"], "typo", 2, 3)
        model = InferenceTagger(
            DummyClassifier(), ["O", "B-PS", "I-PS"], "argmax", 2, 3
        )
        for x, mask in [
            ([[9, 11, 3]], [[1, 1, 1]]),
            ([[2, 11, 9]], [[1, 1, 1]]),
            ([[2, 3, 3]], [[1, 0, 1]]),
        ]:
            with self.assertRaises(ValueError):
                model(torch.tensor(x), torch.tensor(mask))

    def test_historical_char_expansion_metric(self):
        class Tokenizer:
            def convert_ids_to_tokens(self, ids):
                return ["[CLS]", "▁가", "나", "##다", "[SEP]"]

        class Prediction(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                logits = torch.zeros(1, 5, 3)
                logits[0, 1, 1] = 10
                logits[0, 2, 1] = 10
                logits[0, 3, 0] = 10
                return self.anchor.sum(), logits

        ds = TensorDataset(
            torch.tensor([[2, 10, 11, 12, 3]]),
            torch.ones(1, 5, dtype=torch.long),
            torch.tensor([[-100, 1, 2, 0, -100]]),
        )
        result = evaluate(
            DataLoader(ds), Prediction(), Tokenizer(), {0: "O", 1: "B-PS", 2: "I-PS"}
        )
        self.assertEqual(result[2], 0.0)
        self.assertAlmostEqual(result[5], 0.4)


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


class IntegrityTests(unittest.TestCase):
    def test_checkpoint_strictness_and_position_buffer(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.pth"
            config = SimpleNamespace(max_position_embeddings=4)
            torch.save(
                {
                    "weight": torch.ones(2),
                    "bert.embeddings.position_ids": torch.arange(4)[None, :],
                },
                path,
            )
            model, removed = load_checkpoint(TinyModel, config, path)
            self.assertEqual(removed, ["bert.embeddings.position_ids"])
            torch.save(
                {
                    "weight": torch.ones(2),
                    "bert.embeddings.position_ids": torch.ones(1, 4),
                },
                path,
            )
            with self.assertRaises(ValueError):
                load_checkpoint(TinyModel, config, path)
            torch.save({"unrecognized": torch.ones(2)}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(TinyModel, config, path)

    def test_manifest_rejects_data_changes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            files = {name: root / name for name in ["checkpoint", "train", "dev"]}
            for name, path in files.items():
                path.write_text(name)
            m = dict(
                schema_version=1,
                model_name="kobert",
                revision="pinned",
                decoder="legal_bio",
                max_seq_len=50,
                files={n: sha256(p) for n, p in files.items()},
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(m))
            args = argparse.Namespace(
                manifest=str(manifest),
                model_name="kobert",
                revision="pinned",
                decoder="legal_bio",
                max_seq_len=50,
            )
            self.assertEqual(validate_manifest(args, files), m)
            files["dev"].write_text("changed")
            with self.assertRaisesRegex(ValueError, "dev SHA-256"):
                validate_manifest(args, files)

    def test_checks_survive_python_optimization(self):
        script = "from inference import require_equal; require_equal('changed','expected','checksum')"
        result = subprocess.run(
            [sys.executable, "-O", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: checksum", result.stderr)

    def test_result_creation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "nested/result.json"
            write_result(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_result(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            with self.assertRaises(ValueError):
                write_result(Path(root) / "bad.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
