import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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


GAUSSIAN_SOURCE = "gaussian"
INTERPOLATED_SOURCE = "interpolated"

FEATURE_COLUMNS = [
    "보통주자본비율(%)",
    "기본자본비율(%)",
    "총자본비율(%)",
    "단순기본자본비율(%)",
    "공제후자본양수",
    "총자산순이익률ROA(%)",
    "영업흑자",
    "이익잉여금양수",
    "예대율(%)",
    "부채비율(%)",
    "유동비율(%)",
    "자기자본비율(%)",
    "자본총계",
]


def format_integer_like(value):
    if pd.isna(value):
        return value
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune with Gaussian and interpolated data in a 1:2 ratio."
    )
    parser.add_argument(
        "--gaussian_data_path",
        default="data/train/text/train_gaussian.csv",
    )
    parser.add_argument(
        "--interpolated_data_path",
        default="data/train/text/train_interpolated.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/finetuned_xlm-roberta-large",
    )
    parser.add_argument("--model_name", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Batch size. The sampler rounds this to the nearest 1:2 composition.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def format_prompt(row):
    return (
        f"어떤 은행의 보통주자본비율이 {row['보통주자본비율(%)']}%, "
        f"기본자본비율이 {row['기본자본비율(%)']}%, "
        f"총자본비율이 {row['총자본비율(%)']}%, "
        f"단순기본자본비율이 {row['단순기본자본비율(%)']}%, "
        f"공제후자본양수가 {format_integer_like(row['공제후자본양수'])}, "
        f"총자산순이익률ROA가 {row['총자산순이익률ROA(%)']}%, "
        f"영업흑자가 {format_integer_like(row['영업흑자'])}, "
        f"이익잉여금양수가 {format_integer_like(row['이익잉여금양수'])}, "
        f"예대율이 {row['예대율(%)']}%입니다. "
        f"어떤 기업의 부채비율이 {row['부채비율(%)']}%, "
        f"유동비율이 {row['유동비율(%)']}%, "
        f"자기자본비율이 {row['자기자본비율(%)']}%이고, "
        f"자본총계가 {row['자본총계']}입니다. "
        "이 은행은 이 기업에게 대출할 수 있나요?"
    )


def build_text_label_dataframe(raw_df, source):
    df = raw_df[FEATURE_COLUMNS + ["label"]].copy()
    tqdm.pandas(desc=f"Formatting {source}")
    df["text"] = df.progress_apply(format_prompt, axis=1)
    df = df[["text", "label"]]
    df["label"] = df["label"].astype(int)
    df["source"] = source
    return df


def split_one_source(df, eval_size, seed):
    train_df, eval_df = train_test_split(
        df,
        test_size=eval_size,
        random_state=seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def load_mixed_dataset(args):
    gaussian_raw = pd.read_csv(args.gaussian_data_path, low_memory=False)
    interpolated_raw = pd.read_csv(args.interpolated_data_path, low_memory=False)

    gaussian_df = build_text_label_dataframe(gaussian_raw, GAUSSIAN_SOURCE)
    interpolated_df = build_text_label_dataframe(
        interpolated_raw, INTERPOLATED_SOURCE
    )

    gaussian_train, gaussian_eval = split_one_source(gaussian_df, args.eval_size, args.seed)
    interpolated_train, interpolated_eval = split_one_source(
        interpolated_df, args.eval_size, args.seed
    )
    train_df = pd.concat([gaussian_train, interpolated_train], ignore_index=True)
    eval_df = pd.concat([gaussian_eval, interpolated_eval], ignore_index=True)

    return train_df, eval_df


def tokenize_dataframe(df, tokenizer, max_length):
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize_fn, batched=True)


class RatioSampler(Sampler):
    def __init__(self, source_names, batch_size, seed):
        self.gaussian_per_batch = max(1, int(round(batch_size / 3)))
        self.interpolated_per_batch = batch_size - self.gaussian_per_batch
        self.seed = seed
        self.epoch = 0
        self.gaussian_indices = [
            index for index, source in enumerate(source_names) if source == GAUSSIAN_SOURCE
        ]
        self.interpolated_indices = [
            index
            for index, source in enumerate(source_names)
            if source == INTERPOLATED_SOURCE
        ]

        self.num_batches = min(
            len(self.gaussian_indices) // self.gaussian_per_batch,
            len(self.interpolated_indices) // self.interpolated_per_batch,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        gaussian = np.array(self.gaussian_indices, dtype=np.int64)
        interpolated = np.array(self.interpolated_indices, dtype=np.int64)
        rng.shuffle(gaussian)
        rng.shuffle(interpolated)

        for batch_idx in range(self.num_batches):
            g_start = batch_idx * self.gaussian_per_batch
            i_start = batch_idx * self.interpolated_per_batch
            batch = np.concatenate(
                [
                    gaussian[g_start : g_start + self.gaussian_per_batch],
                    interpolated[
                        i_start : i_start + self.interpolated_per_batch
                    ],
                ]
            )
            rng.shuffle(batch)
            yield from batch.tolist()

    def __len__(self):
        return self.num_batches * (
            self.gaussian_per_batch + self.interpolated_per_batch
        )


class RatioTrainer(Trainer):
    def __init__(self, *args, train_source_names, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_source_names = train_source_names

    def _get_train_sampler(self):
        return RatioSampler(
            self.train_source_names,
            self._train_batch_size,
            self.args.seed,
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def build_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        seed=args.seed,
        evaluation_strategy="epoch",
    )


def main():
    args = parse_args()

    if args.train_batch_size < 3:
        raise ValueError("--train_batch_size must be at least 3.")

    set_seed(args.seed)
    train_df, eval_df = load_mixed_dataset(args)
    sampler = RatioSampler(train_df["source"].tolist(), args.train_batch_size, args.seed)

    print(f"train rows: {len(train_df):,}")
    print(f"eval rows: {len(eval_df):,}")
    print(
        f"batch ratio: {sampler.gaussian_per_batch} gaussian + "
        f"{sampler.interpolated_per_batch} interpolated"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    train_dataset = tokenize_dataframe(train_df, tokenizer, args.max_length)
    eval_dataset = tokenize_dataframe(eval_df, tokenizer, args.max_length)

    trainer = RatioTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        train_source_names=train_df["source"].tolist(),
    )

    trainer.train()

    best_dir = Path(args.output_dir) / "best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
    print("\nFinal eval metrics")
    print(metrics)
    pd.DataFrame([metrics]).to_csv(Path(args.output_dir) / "eval_metrics.csv", index=False)


if __name__ == "__main__":
    main()
