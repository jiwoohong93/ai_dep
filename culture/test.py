"""Evaluate culture NER using the repository's historical metric.

Validate inputs, load a checkpoint, decode BIO predictions and save metrics.
Use --decoder legal_bio with --manifest culture/certification.json for the
certification configuration; the general CLI retains argmax as its default.
"""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from dataset import get_dataset
from models import get_model
from inference import (
    KOBERT_REVISION,
    InferenceTagger,
    load_checkpoint,
    require_equal,
    sha256,
    write_result,
)


def evaluate(dataloader_valid, model, tokenizer, id2label, echo_num=40):
    """Return entity and expanded-token-string precision, recall and micro F1."""

    total_step_per_epoch = len(dataloader_valid)
    total_loss = 0.0
    in_token_ids = None
    preds = None
    out_label_ids = None

    model.eval()
    for batch in tqdm(dataloader_valid, desc="KLUE-NER dev", mininterval=15):
        batch = tuple(t.to(next(model.parameters()).device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            in_token_ids = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
            in_token_ids = np.append(
                in_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0
            )

    loss = total_loss / total_step_per_epoch
    preds = np.argmax(preds, axis=2)

    gt_token_label_list = [[] for _ in range(out_label_ids.shape[0])]
    pred_token_label_list = [[] for _ in range(out_label_ids.shape[0])]
    gt_char_label_list = [[] for _ in range(out_label_ids.shape[0])]
    pred_char_label_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    for i in range(out_label_ids.shape[0]):  # sentence
        token_ids = in_token_ids[i]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        for j in range(out_label_ids.shape[1]):  # token
            if out_label_ids[i, j] == pad_token_label_id:
                continue

            gt_label_id = out_label_ids[i][j]
            gt_label = id2label[gt_label_id]
            gt_token_label_list[i].append(gt_label)
            pred_label_id = preds[i][j]
            pred_label = id2label[pred_label_id]
            pred_token_label_list[i].append(pred_label)

            token = tokens[j]
            token = token.replace("##", "")
            if token[0] == "[" and token[-1] == "]":
                gt_char_label_list[i].append(gt_label)
                pred_char_label_list[i].append(pred_label)
            else:
                gt_char_label_list[i] += [gt_label] * len(token)
                pred_char_label_list[i] += [pred_label] * len(token)

    result = classification_report(gt_token_label_list, pred_token_label_list)
    print("[entity f1 score]")
    print(result)

    return (
        precision_score(gt_token_label_list, pred_token_label_list),
        recall_score(gt_token_label_list, pred_token_label_list),
        f1_score(gt_token_label_list, pred_token_label_list),
        precision_score(gt_char_label_list, pred_char_label_list),
        recall_score(gt_char_label_list, pred_char_label_list),
        f1_score(gt_char_label_list, pred_char_label_list),
    )


def validate_manifest(args, files):
    """Check the approved inputs before reading model weights, also with python -O."""
    if args.manifest is None:
        return None
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    require_equal(manifest["schema_version"], 1, "Manifest schema")
    for field in ["model_name", "revision", "decoder", "max_seq_len"]:
        require_equal(getattr(args, field), manifest[field], field)
    for name in ["checkpoint", "train", "dev"]:
        require_equal(sha256(files[name]), manifest["files"][name], f"{name} SHA-256")
    return manifest


def main(args):
    """Run evaluation with the requested decoder and record reproducibility data."""
    started = time.monotonic()
    if args.max_seq_len < 2 or args.bs < 1 or args.threads < 1:
        raise ValueError("max_seq_len must be >=2; bs and threads must be positive")
    if args.output_json and Path(args.output_json).exists():
        raise FileExistsError(f"Output exists; choose a new path: {args.output_json}")
    files = {
        "checkpoint": Path(args.model_path),
        "train": Path(args.dataset_path) / "klue-ner-v1.1_train.tsv",
        "dev": Path(args.dataset_path) / "klue-ner-v1.1_dev.tsv",
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} file: {path}; prepare files before evaluation"
            )
    if args.model_name == "kobert" and args.revision is None:
        args.revision = KOBERT_REVISION
    manifest = validate_manifest(args, files)
    input_hashes = {name: sha256(path) for name, path in files.items()}
    torch.set_num_threads(args.threads)
    ModelConfig, Tokenizer, Model, pretrain_name = get_model(args.model_name)
    asset_args = dict(revision=args.revision, local_files_only=args.offline)
    tokenizer = Tokenizer.from_pretrained(
        pretrain_name,
        do_lower_case=False,
        trust_remote_code=args.model_name == "kobert",
        **asset_args,
    )
    dataset_train, dataset_valid = get_dataset(
        args.dataset_path, tokenizer, args.max_seq_len
    )
    labels = dataset_train.labels
    if manifest is not None:
        require_equal(labels, manifest["labels"], "Checkpoint label order")
    if not any(y != -100 for row in dataset_valid.list_of_label_ids for y in row):
        raise ValueError("Evaluation data contains no scored tokens")
    id2label = dict(enumerate(labels))
    config = ModelConfig.from_pretrained(
        pretrain_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id={label: i for i, label in id2label.items()},
        **asset_args,
    )
    model, removed_buffers = load_checkpoint(Model, config, args.model_path)
    if args.max_seq_len > config.max_position_embeddings:
        raise ValueError("max_seq_len exceeds the model's position limit")
    tagger = InferenceTagger(
        model, labels, args.decoder, tokenizer.cls_token_id, tokenizer.sep_token_id
    )
    tagger.to(args.device).eval()
    values = evaluate(
        DataLoader(dataset_valid, batch_size=args.bs, shuffle=False),
        tagger,
        tokenizer,
        id2label,
    )
    metrics = {
        "entity": dict(zip(["precision", "recall", "f1"], map(float, values[:3]))),
        "char": dict(zip(["precision", "recall", "f1"], map(float, values[3:]))),
    }
    # Detect file replacement during the run instead of reporting a stale hash.
    for name, path in files.items():
        require_equal(
            sha256(path), input_hashes[name], f"{name} changed during evaluation"
        )
    target = manifest["target_char_f1"] if manifest else 0.94
    scored_records = sum(
        any(y != -100 for y in row) for row in dataset_valid.list_of_label_ids
    )
    source_names = [
        "test.py",
        "dataset.py",
        "models.py",
        "inference.py",
        "legacy_decode.py",
    ]
    result = dict(
        metrics=metrics,
        target_char_f1=target,
        target_met=metrics["char"]["f1"] >= target,
        decoder=args.decoder,
        model_name=args.model_name,
        revision=args.revision,
        max_seq_len=args.max_seq_len,
        labels=labels,
        train_parsed_records=len(dataset_train),
        dev_parsed_records=len(dataset_valid),
        dev_nonempty_sentences=scored_records,
        removed_legacy_buffers=removed_buffers,
        files={
            name: dict(path=str(path.resolve()), sha256=input_hashes[name])
            for name, path in files.items()
        },
        source_sha256={
            name: sha256(Path(__file__).with_name(name)) for name in source_names
        },
        manifest_sha256=sha256(args.manifest) if args.manifest else None,
        certification_inputs_verified=manifest is not None,
        device=args.device,
        evaluation_seconds=time.monotonic() - started,
        environment=dict(
            python=platform.python_version(),
            **{
                name: version(name)
                for name in [
                    "torch",
                    "transformers",
                    "seqeval",
                    "sentencepiece",
                    "numpy",
                ]
            },
        ),
        metric="original culture/test.py token-string-expanded seqeval Char F1",
        additional_training_in_this_run=False,
        scope="Preserves gold-dependent segmentation, same-type merging and truncation. "
        "Not raw-character macro F1, a hidden-test result, or an accreditation certificate.",
        selection_note=(
            manifest["selection_note"]
            if manifest
            else "Checkpoint training/selection history not verified."
        ),
    )
    for kind, values in metrics.items():
        print(
            f"{kind}: precision={values['precision']:.10f}, recall={values['recall']:.10f}, f1={values['f1']:.10f}"
        )
    print(
        f"Char F1 (%): {metrics['char']['f1'] * 100:.10f}; target_met={result['target_met']}"
    )
    if args.output_json:
        write_result(args.output_json, result)
    return result


def parse_args():
    """Parse data paths, model settings, device and result-file options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Directory containing official train/dev TSV files",
    )
    parser.add_argument(
        "--model_path", required=True, help="Plain PyTorch state_dict checkpoint"
    )
    parser.add_argument(
        "--model_name",
        default="koelectra",
        choices=["koelectra-v3", "koelectra", "kcbert", "kobert"],
    )
    parser.add_argument("--decoder", choices=["argmax", "legal_bio"], default="argmax")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--revision",
        default=None,
        help="KoBERT defaults to the pinned certification revision",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use existing model/tokenizer cache only"
    )
    parser.add_argument(
        "--manifest", help="Verify model settings, input SHA-256 values and label order"
    )
    parser.add_argument(
        "--output_json",
        help="Create a result file; an existing file will never be overwritten",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
