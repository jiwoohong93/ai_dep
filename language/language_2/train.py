#!/usr/bin/env python3
"""KOLD 단일 task fine-tuning (논문 Table 5, 6).

task:
    A         offensiveness (binary, 전체)            -> binary P/R/F1 (positive=OFF)
    B         target type 4-class (OFF=True)          -> macro P/R/F1
    C         target group 세부 27-class (GRP)        -> macro P/R/F1  (논문 Table 5 Level C)
    C_attr    target group attribute 5-class (GRP)    -> macro P/R/F1  (참고용)
    off_span  offensive span BIO (OFF=True)           -> char-level P/R/F1
    tgt_span  target span BIO (OFF=True)              -> char-level P/R/F1

입력은 (title, comment) 문장쌍. span 라벨은 comment 토큰에만 붙는다.
학습 후 dev/test 점수를 --out 디렉토리의 metrics.json 에 저장한다 (체크포인트는 저장하지 않음).

실행 예:
    CUDA_VISIBLE_DEVICES=0 python language_2/train.py --task A --model klue/bert-base \
        --lr 2e-5 --epochs 3 --seed 42 --out language_2/runs/A/bert-base/lr2e-5_ep3_s42
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          AutoTokenizer, DataCollatorForTokenClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_LEN = 160  # title+comment 최대 146 토큰

CLS_LABELS = {
    "A": ["NOT", "OFF"],
    "B": ["UNT", "IND", "OTH", "GRP"],
    "C_attr": ["GEN", "RACE", "POL", "REL", "MISC"],
}
LABEL_KEY = {"A": "off", "B": "tgt", "C": "grp_fine", "C_attr": "grp"}
BIO = ["O", "B", "I"]


def load(split, task):
    rows = [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    if task in ("B", "off_span", "tgt_span"):
        rows = [r for r in rows if r["off"]]
    elif task in ("C", "C_attr"):
        rows = [r for r in rows if r["grp"]]
    return rows


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ---------------------------------------------------------------------------
# sequence classification
# ---------------------------------------------------------------------------
def encode_cls(rows, tok, task, label_names):
    enc = tok([r["title"] for r in rows], [r["comment"] for r in rows],
              truncation=True, max_length=MAX_LEN)
    labels = [label_names.index(r[LABEL_KEY[task]]) if task != "A" else r["off"] for r in rows]
    return ListDataset([{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                         **({"token_type_ids": enc["token_type_ids"][i]} if "token_type_ids" in enc else {}),
                         "labels": labels[i]} for i in range(len(rows))]), labels


def cls_metrics(task, labels, preds):
    if task == "A":
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    else:
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = float(np.mean(np.array(labels) == np.array(preds)))
    return {"precision": 100 * p, "recall": 100 * r, "f1": 100 * f, "accuracy": 100 * acc}


# ---------------------------------------------------------------------------
# span prediction (BIO token classification)
# ---------------------------------------------------------------------------
def encode_span(rows, tok, span_key):
    items, metas = [], []
    for r in rows:
        enc = tok(r["title"], r["comment"], truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        seq_ids = enc.sequence_ids()
        gold = set()
        for s, e in r[span_key]:
            gold.update(range(s, e))
        labels, prev_in = [], False
        comment_tokens = []
        for ti, ((s, e), sid) in enumerate(zip(enc["offset_mapping"], seq_ids)):
            if sid != 1 or s == e:
                labels.append(-100)
                prev_in = False
                continue
            comment_tokens.append((ti, s, e))
            inside = any(c in gold for c in range(s, e))
            # 직전 토큰과 gold span 이 이어지면 I, 아니면 B
            contiguous = prev_in and all(c in gold for c in range(comment_tokens[-2][2], s)) \
                if len(comment_tokens) > 1 else False
            labels.append(0 if not inside else (2 if contiguous else 1))
            prev_in = inside
        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        items.append(item)
        metas.append({"tokens": comment_tokens, "gold": gold})
    return ListDataset(items), metas


def decode_span(pred_ids, tokens):
    """B/I 토큰을 span 으로 묶고, 같은 span 안의 토큰 사이 공백도 포함한 문자 집합을 반환."""
    chars, cur = set(), None
    for ti, s, e in tokens:
        lab = pred_ids[ti]
        if lab == 1 or (lab == 2 and cur is None):
            cur = [s, e]
        elif lab == 2:
            cur[1] = e
        else:
            if cur:
                chars.update(range(*cur))
            cur = None
            continue
        chars.update(range(*cur))
    return chars


def span_metrics(metas, preds):
    """Da San Martino et al. (2019) / SemEval-2021 Task 5 방식의 문자 단위 P/R/F1 (샘플 평균).
    gold, pred 모두 비면 1, 한쪽만 비면 0."""
    P, R, F = [], [], []
    for m, pred in zip(metas, preds):
        g = m["gold"]
        if not g and not pred:
            p = r = f = 1.0
        elif not g or not pred:
            p = r = f = 0.0
        else:
            tp = len(g & pred)
            p, r = tp / len(pred), tp / len(g)
            f = 2 * p * r / (p + r) if tp else 0.0
        P.append(p), R.append(r), F.append(f)
    return {"precision": 100 * np.mean(P), "recall": 100 * np.mean(R), "f1": 100 * np.mean(F)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["A", "B", "C", "C_attr", "off_span", "tgt_span"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save_model", action="store_true", help="학습된 모델/토크나이저를 OUT/model 에 저장 (데모용)")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    is_span = args.task.endswith("_span")
    splits = {s: load(s, args.task) for s in ("train", "dev", "test")}

    if is_span:
        span_key = "off_spans" if args.task == "off_span" else "tgt_spans"
        data = {s: encode_span(rows, tok, span_key) for s, rows in splits.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(BIO), id2label=dict(enumerate(BIO)), label2id={l: i for i, l in enumerate(BIO)})
        collator = DataCollatorForTokenClassification(tok)
    else:
        # 세부 group 은 train 에 등장한 라벨 전체 (27개)
        labels = CLS_LABELS.get(args.task) or sorted({r["grp_fine"] for r in splits["train"]})
        data = {s: encode_cls(rows, tok, args.task, labels) for s, rows in splits.items()}
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
        collator = DataCollatorWithPadding(tok)

    args.out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out / "tmp"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        bf16=True,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=100,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=data["train"][0], data_collator=collator)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "n": {s: len(rows) for s, rows in splits.items()}, "train_time_sec": train_time}
    for s in ("dev", "test"):
        logits = trainer.predict(data[s][0]).predictions
        pred = logits.argmax(-1)
        if is_span:
            metas = data[s][1]
            preds = [decode_span(pred[i], m["tokens"]) for i, m in enumerate(metas)]
            result[s] = span_metrics(metas, preds)
            # 참고용: gold span 이 있는 샘플만으로 계산 (빈 gold 채점 규칙의 영향 확인)
            keep = [i for i, m in enumerate(metas) if m["gold"]]
            result[s + "_nonempty_gold"] = {**span_metrics([metas[i] for i in keep], [preds[i] for i in keep]),
                                            "n": len(keep)}
        else:
            result[s] = cls_metrics(args.task, data[s][1], pred.tolist())
    if args.save_model:
        trainer.save_model(str(args.out / "model"))
        tok.save_pretrained(str(args.out / "model"))
    # metrics.json 은 마지막에 쓴다 (sweep 이 완료 여부를 이 파일로 판단)
    json.dump(result, open(args.out / "metrics.json", "w"), indent=2)
    print(json.dumps({k: result[k] for k in ("dev", "test")}, indent=2))


if __name__ == "__main__":
    main()
