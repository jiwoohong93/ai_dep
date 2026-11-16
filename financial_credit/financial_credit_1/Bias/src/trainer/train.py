"""Adult 프롬프트 분류기 학습 — Regulation train.py 이식 (단일 데이터 원천 버전).

Regulation과 동일 구조: FacebookAI/xlm-roberta-large + sequence classification
(num_labels=2), max_length 256, lr 1e-5. RatioSampler(합성 2원천 혼합)는 실데이터
단일 원천이라 제거.

프로토콜(KAIST baseline과 대칭): adult.data 32,561 전체로 학습, val 없음,
매 에폭 체크포인트 저장(epoch-1, epoch-2, ...) → 학습 후 src/models/test.py를
에폭별로 돌려 test 지표의 범위/최종에폭을 보고.

--gaussian_path를 주면 rule 경계 gaussian 생성데이터를 Regulation 레시피대로
배치 내 실데이터:생성 = 2:1로 혼합해 학습한다 (RatioSampler 포팅).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Sampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REAL_SOURCE = "real"
GAUSSIAN_SOURCE = "gaussian"

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import format_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R on Adult prompts.")
    parser.add_argument("--train_path", default=str(BIAS_ROOT / "data/train/text/train.csv"))
    parser.add_argument("--gaussian_path", default=None,
                        help="rule 경계 gaussian csv — 주면 배치 2:1(실:생성) 혼합")
    parser.add_argument("--output_dir", default=str(BIAS_ROOT / "outputs/finetuned_xlm-roberta-large"))
    parser.add_argument("--model_name", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_text_dataframe(path, desc):
    df = pd.read_csv(path)
    tqdm.pandas(desc=f"Formatting {desc}")
    out = pd.DataFrame({
        "text": df.progress_apply(format_prompt, axis=1),
        "label": df["label"].astype(int),
    })
    return out


def tokenize_dataframe(df, tokenizer, max_length):
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


class RatioSampler(Sampler):
    """배치마다 gaussian:real = 1:2 구성 (Regulation train.py 포팅)."""

    def __init__(self, source_names, batch_size, seed):
        self.gaussian_per_batch = max(1, int(round(batch_size / 3)))
        self.real_per_batch = batch_size - self.gaussian_per_batch
        self.seed = seed
        self.epoch = 0
        self.gaussian_indices = [
            i for i, s in enumerate(source_names) if s == GAUSSIAN_SOURCE
        ]
        self.real_indices = [
            i for i, s in enumerate(source_names) if s == REAL_SOURCE
        ]
        self.num_batches = min(
            len(self.gaussian_indices) // self.gaussian_per_batch,
            len(self.real_indices) // self.real_per_batch,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        gaussian = np.array(self.gaussian_indices, dtype=np.int64)
        real = np.array(self.real_indices, dtype=np.int64)
        rng.shuffle(gaussian)
        rng.shuffle(real)
        for b in range(self.num_batches):
            batch = np.concatenate([
                gaussian[b * self.gaussian_per_batch : (b + 1) * self.gaussian_per_batch],
                real[b * self.real_per_batch : (b + 1) * self.real_per_batch],
            ])
            rng.shuffle(batch)
            yield from batch.tolist()

    def __len__(self):
        return self.num_batches * (self.gaussian_per_batch + self.real_per_batch)


class RatioTrainer(Trainer):
    def __init__(self, *args, train_source_names, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_source_names = train_source_names

    def _get_train_sampler(self):
        return RatioSampler(
            self.train_source_names, self._train_batch_size, self.args.seed
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    train_df = load_text_dataframe(args.train_path, "train")
    train_df["source"] = REAL_SOURCE
    if args.gaussian_path:
        gaussian_df = load_text_dataframe(args.gaussian_path, "gaussian")
        gaussian_df["source"] = GAUSSIAN_SOURCE
        train_df = pd.concat([train_df, gaussian_df], ignore_index=True)
        print(f"gaussian rows: {len(gaussian_df):,} (배치 실:생성 = 2:1 혼합)")
    print(f"train rows: {len(train_df):,} (val 없음 — KAIST 프로토콜 대칭)")
    print(f"example prompt:\n{train_df['text'].iloc[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    train_dataset = tokenize_dataframe(train_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",   # 매 에폭 checkpoint-* 저장 → 사후 에폭별 test 측정
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=None,
        fp16=False,
        report_to="none",
        seed=args.seed,
        evaluation_strategy="no",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if args.gaussian_path:
        trainer = RatioTrainer(
            **trainer_kwargs, train_source_names=train_df["source"].tolist()
        )
    else:
        trainer = Trainer(**trainer_kwargs)
    trainer.train()

    # 마지막 에폭 모델을 best/에 저장 (에폭별 비교 후 대표 체크포인트는 사후 결정)
    best_dir = Path(args.output_dir) / "best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"final-epoch model saved to {best_dir}")
    print(f"per-epoch checkpoints: {args.output_dir}/checkpoint-*")


if __name__ == "__main__":
    main()
