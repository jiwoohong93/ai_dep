#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_header(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def clean_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def read_sheet_selected(
    xlsx_path: Path,
    sheet_name: str,
    needed_columns: list[str],
    required_columns: list[str],
    scan_rows: int = 12,
) -> pd.DataFrame:
    log(f"Reading {xlsx_path.name} / {sheet_name}")
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    normalized_needed = [normalize_header(c) for c in needed_columns]
    normalized_required = {normalize_header(c) for c in required_columns}

    header_row = None
    header_map = {}
    for row_no, row in enumerate(ws.iter_rows(values_only=True), start=1):
        normalized = [normalize_header(v) for v in row]
        present = set(v for v in normalized if v)
        if normalized_required.issubset(present):
            header_row = row_no
            for idx, name in enumerate(normalized):
                if name and name not in header_map:
                    header_map[name] = idx
            break
        if row_no >= scan_rows:
            break

    if header_row is None:
        raise ValueError(f"Could not find header row in {xlsx_path} / {sheet_name}")

    missing = [c for c in normalized_required if c not in header_map]
    if missing:
        raise ValueError(f"Missing required columns in {sheet_name}: {missing}")

    selected = [(name, header_map.get(name)) for name in normalized_needed]
    rows = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        out = {}
        any_value = False
        for original_name, normalized_name in zip(needed_columns, normalized_needed):
            idx = header_map.get(normalized_name)
            value = row[idx] if idx is not None and idx < len(row) else None
            value = clean_value(value)
            out[normalized_name] = value
            any_value = any_value or value is not None
        if any_value:
            rows.append(out)
    wb.close()

    df = pd.DataFrame(rows, columns=normalized_needed)
    log(f"Loaded {sheet_name}: {len(df):,} rows, {len(df.columns)} columns")
    return df


def join_unique(values: Iterable, max_items: int = 8, max_chars: int = 1800) -> str | None:
    seen = set()
    parts = []
    for value in values:
        value = clean_value(value)
        if value is None:
            continue
        key = value.strip()
        if key in seen:
            continue
        seen.add(key)
        parts.append(key)
        if len(parts) >= max_items:
            break
    if not parts:
        return None
    text = ", ".join(parts)
    return text[:max_chars]


def compose_text(prefix_pairs: list[tuple[str, str | None]], max_chars: int = 4000) -> str:
    parts = []
    for label, value in prefix_pairs:
        value = clean_value(value)
        if value is not None:
            parts.append(f"{label}: {value}")
    return "; ".join(parts)[:max_chars]


def build_applicants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["이력서번호", "지원자"]).copy()
    agg_cols = [c for c in df.columns if c not in {"이력서번호", "지원자"}]
    grouped = (
        df.groupby(["이력서번호", "지원자"], sort=False)[agg_cols]
        .agg(lambda s: join_unique(s, max_items=6, max_chars=1600))
        .reset_index()
    )
    grouped["text"] = grouped.apply(
        lambda r: compose_text(
            [
                ("희망직종", r.get("희망(경력)직종1")),
                ("희망세부직종", r.get("희망(경력)세부직종1")),
                ("추가희망직종", r.get("희망(경력)직종2")),
                ("추가희망세부직종", r.get("희망(경력)세부직종2")),
                ("직무태그", r.get("직무태그")),
                ("희망업종", r.get("희망(경력)업종1")),
                ("희망세부업종", r.get("희망(경력)세부업종1")),
                ("전체경력", r.get("전체경력")),
                ("근무형태", r.get("근무형태")),
                ("희망연봉", r.get("희망연봉")),
                ("최종연봉", r.get("최종연봉")),
                ("희망근무지역", r.get("근무지역")),
                ("최종학력", r.get("최종학력")),
                ("학교명", r.get("학교명")),
                ("학력구분", r.get("학력구분")),
                ("전공", r.get("전공")),
                ("계열", r.get("계열")),
                ("경력업종", r.get("업종")),
                ("경력세부업종", r.get("세부업종")),
                ("회사명", r.get("회사명")),
                ("직위", r.get("직위")),
                ("담당업무", r.get("담당업무")),
                ("프로젝트", r.get("프로젝트명")),
                ("내용및성과", r.get("내용및성과")),
                ("언어", r.get("언어")),
                ("공인시험", r.get("공인시험")),
                ("점수", r.get("점수")),
                ("자격증", r.get("자격증명")),
            ]
        ),
        axis=1,
    )
    grouped = grouped[grouped["text"].str.len() > 0].copy()
    grouped = grouped.rename(columns={"이력서번호": "resume_id", "지원자": "applicant_id"})
    return grouped[["resume_id", "applicant_id", "text"]]


def build_jobs(job_df: pd.DataFrame, company_df: pd.DataFrame) -> pd.DataFrame:
    job_df = job_df.dropna(subset=["기업", "공고번호"]).copy()
    company_small = company_df.dropna(subset=["기업"]).drop_duplicates("기업").copy()
    company_small = company_small.rename(
        columns={
            "기업": "기업",
            "기업명": "기업명",
            "기업형태": "기업형태",
            "자본금": "자본금",
            "매출현황": "매출현황",
            "사원수": "사원수",
        }
    )
    job_df = job_df.merge(company_small, on="기업", how="left", suffixes=("", "_기업"))
    agg_cols = [c for c in job_df.columns if c not in {"기업", "공고번호"}]
    grouped = (
        job_df.groupby(["기업", "공고번호"], sort=False)[agg_cols]
        .agg(lambda s: join_unique(s, max_items=4, max_chars=1200))
        .reset_index()
    )
    grouped["text"] = grouped.apply(
        lambda r: compose_text(
            [
                ("공고명", r.get("공고명")),
                ("기업명", r.get("기업명")),
                ("업종", r.get("업종")),
                ("세부업종", r.get("세부업종")),
                ("기업형태", r.get("기업형태")),
                ("자본금", r.get("자본금")),
                ("매출현황", r.get("매출현황")),
                ("사원수", r.get("사원수")),
                ("직종1", r.get("직종1")),
                ("세부직종1", r.get("세부직종1")),
                ("직종2", r.get("직종2")),
                ("세부직종2", r.get("세부직종2")),
                ("직종3", r.get("직종3")),
                ("세부직종3", r.get("세부직종3")),
                ("키워드", r.get("키워드")),
                ("근무형태", r.get("근무형태")),
                ("채용직급", r.get("채용직급")),
                ("모집인원", r.get("모집인원")),
                ("나이", r.get("나이")),
                ("경력", r.get("경력")),
                ("급여조건", r.get("급여조건")),
                ("학력", r.get("학력")),
                ("접수기간", r.get("접수기간")),
                ("근무지역", r.get("근무지역")),
            ]
        ),
        axis=1,
    )
    grouped = grouped[grouped["text"].str.len() > 0].copy()
    grouped = grouped.rename(columns={"기업": "company_id", "공고번호": "job_id"})
    return grouped[["company_id", "job_id", "text"]]


def build_samples(
    applications: pd.DataFrame,
    applicants: pd.DataFrame,
    jobs: pd.DataFrame,
    neg_ratio: int,
    seed: int,
) -> pd.DataFrame:
    applications = applications.rename(
        columns={"이력서번호": "resume_id", "지원자": "applicant_id", "기업": "company_id", "공고번호": "job_id"}
    )
    applications = applications.dropna(subset=["resume_id", "applicant_id", "company_id", "job_id"])
    applications = applications.drop_duplicates(["resume_id", "job_id"]).copy()
    valid_resumes = set(applicants["resume_id"])
    valid_jobs = set(jobs["job_id"])
    applications = applications[
        applications["resume_id"].isin(valid_resumes) & applications["job_id"].isin(valid_jobs)
    ].copy()
    log(f"Positive application pairs after text availability filter: {len(applications):,}")

    rng = np.random.default_rng(seed)
    job_ids = jobs["job_id"].to_numpy()
    job_to_company = dict(zip(jobs["job_id"], jobs["company_id"]))
    positives_by_resume = defaultdict(set)
    for resume_id, job_id in applications[["resume_id", "job_id"]].itertuples(index=False):
        positives_by_resume[resume_id].add(job_id)

    gender_by_applicant = {
        applicant_id: int(g)
        for applicant_id, g in zip(
            applicants["applicant_id"].drop_duplicates(),
            rng.integers(0, 2, size=applicants["applicant_id"].nunique()),
        )
    }
    resume_to_applicant = dict(zip(applicants["resume_id"], applicants["applicant_id"]))

    rows = []
    for resume_id, applicant_id, company_id, job_id in applications[
        ["resume_id", "applicant_id", "company_id", "job_id"]
    ].itertuples(index=False):
        gender = gender_by_applicant.get(applicant_id, 0)
        rows.append((resume_id, applicant_id, company_id, job_id, 1, gender))
        positive_jobs = positives_by_resume[resume_id]
        for _ in range(neg_ratio):
            neg_job = rng.choice(job_ids)
            tries = 0
            while neg_job in positive_jobs and tries < 200:
                neg_job = rng.choice(job_ids)
                tries += 1
            neg_company = job_to_company.get(neg_job)
            rows.append((resume_id, applicant_id, neg_company, neg_job, 0, gender))

    samples = pd.DataFrame(
        rows, columns=["resume_id", "applicant_id", "company_id", "job_id", "label", "gender"]
    )
    labels = samples["label"].to_numpy()
    idx = np.arange(len(samples))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=labels
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    split = np.empty(len(samples), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    samples["split"] = split
    return samples


def prepare(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    applicants_path = out_dir / "applicants.parquet"
    jobs_path = out_dir / "jobs.parquet"
    applications_path = out_dir / "applications.parquet"
    samples_path = out_dir / "samples.parquet"
    summary_path = out_dir / "prepare_summary.json"
    if all(p.exists() for p in [applicants_path, jobs_path, applications_path, samples_path]) and not args.force_prepare:
        log("Prepared parquet files already exist; skipping prepare.")
        return json.loads(summary_path.read_text()) if summary_path.exists() else {}

    data_dir = Path(args.data_dir)
    company_book = data_dir / "스카우트_기업회원DB_20221222.xlsx"
    applicant_book = data_dir / "스카우트_이력사항DB_20221222.xlsx"
    application_book = data_dir / "스카우트_지원내역DB_20221222.xlsx"

    company_cols = ["기업", "기업명", "업종", "세부업종", "자본금", "기업형태", "매출현황", "사원수"]
    job_cols = [
        "기업",
        "공고번호",
        "공고명",
        "업종",
        "세부업종",
        "직종1",
        "세부직종1",
        "직종2",
        "세부직종2",
        "직종3",
        "세부직종3",
        "키워드",
        "근무형태",
        "채용직급",
        "모집인원",
        "나이",
        "경력",
        "급여조건",
        "학력",
        "접수기간",
        "근무지역",
    ]
    applicant_cols = [
        "이력서번호",
        "지원자",
        "희망(경력)직종1",
        "희망(경력)세부직종1",
        "희망(경력)직종2",
        "희망(경력)세부직종2",
        "직무태그",
        "희망(경력)업종1",
        "희망(경력)세부업종1",
        "전체경력",
        "근무형태",
        "희망연봉",
        "최종연봉",
        "근무지역",
        "최종학력",
        "학교명",
        "학력구분",
        "전공",
        "계열",
        "업종",
        "세부업종",
        "회사명",
        "직위",
        "담당업무",
        "프로젝트명",
        "내용및성과",
        "언어",
        "공인시험",
        "점수",
        "자격증명",
    ]
    application_cols = ["이력서번호", "지원자", "기업", "공고번호"]

    company_df = read_sheet_selected(company_book, "기업정보", company_cols, ["기업"])
    job_df = read_sheet_selected(company_book, "공고정보", job_cols, ["기업", "공고번호"])
    applicant_df = read_sheet_selected(applicant_book, "통합정보", applicant_cols, ["이력서번호", "지원자"])
    application_df = read_sheet_selected(
        application_book, "지원내역", application_cols, ["이력서번호", "지원자", "기업", "공고번호"]
    )

    applicants = build_applicants(applicant_df)
    jobs = build_jobs(job_df, company_df)
    applications = application_df.rename(
        columns={"이력서번호": "resume_id", "지원자": "applicant_id", "기업": "company_id", "공고번호": "job_id"}
    )
    samples = build_samples(application_df, applicants, jobs, args.neg_ratio, args.seed)

    applicants.to_parquet(applicants_path, index=False)
    jobs.to_parquet(jobs_path, index=False)
    applications.to_parquet(applications_path, index=False)
    samples.to_parquet(samples_path, index=False)
    summary = {
        "applicants": int(len(applicants)),
        "jobs": int(len(jobs)),
        "applications_raw": int(len(application_df)),
        "applications_named": int(len(applications)),
        "samples": int(len(samples)),
        "samples_by_split": {k: int(v) for k, v in samples["split"].value_counts().to_dict().items()},
        "positive_samples": int((samples["label"] == 1).sum()),
        "negative_samples": int((samples["label"] == 0).sum()),
        "gender_counts": {str(k): int(v) for k, v in samples["gender"].value_counts().to_dict().items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"Prepare summary: {json.dumps(summary, ensure_ascii=False)}")
    del company_df, job_df, applicant_df, application_df, applicants, jobs, applications, samples
    gc.collect()
    return summary


def encode_table(args, table_name: str, id_col: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    out_dir = Path(args.out_dir)
    table_path = out_dir / f"{table_name}.parquet"
    emb_path = out_dir / f"{table_name}_embeddings.npy"
    if emb_path.exists() and not args.force_embed:
        log(f"{emb_path.name} already exists; skipping embedding.")
        return

    df = pd.read_parquet(table_path, columns=[id_col, "text"])
    texts = df["text"].fillna("").astype(str).tolist()
    log(f"Encoding {table_name}: {len(texts):,} texts with {args.embedding_model}")

    model_kwargs = {}
    if torch.cuda.is_available() and args.fp16_embed:
        model_kwargs["torch_dtype"] = torch.float16
    model = SentenceTransformer(
        args.embedding_model,
        revision=args.embedding_revision,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.use_multi_gpu_embed:
        devices = [f"cuda:{i}" for i in range(n_gpus)]
        log(f"Using multi-GPU embedding devices: {devices}, batch_size={args.embed_batch_size}")
        pool = model.start_multi_process_pool(target_devices=devices)
        try:
            embeddings = model.encode_multi_process(
                texts,
                pool,
                batch_size=args.embed_batch_size,
                chunk_size=args.embed_chunk_size,
                show_progress_bar=True,
                precision="float32",
                normalize_embeddings=True,
            )
        finally:
            model.stop_multi_process_pool(pool)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        embeddings = model.encode(
            texts,
            batch_size=args.embed_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
    embeddings = np.asarray(embeddings, dtype=np.float16)
    np.save(emb_path, embeddings)
    log(f"Saved {emb_path} shape={embeddings.shape} dtype={embeddings.dtype}")
    del model, embeddings, texts, df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed(args) -> None:
    encode_table(args, "applicants", "resume_id")
    encode_table(args, "jobs", "job_id")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset:
    def __init__(
        self,
        sample_indices: np.ndarray,
        resume_indices: np.ndarray,
        job_indices: np.ndarray,
        labels: np.ndarray,
        genders: np.ndarray,
        applicant_embeddings: np.ndarray,
        job_embeddings: np.ndarray,
    ):
        self.sample_indices = sample_indices.astype(np.int64)
        self.resume_indices = resume_indices.astype(np.int64)
        self.job_indices = job_indices.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.genders = genders.astype(np.float32)
        self.applicant_embeddings = applicant_embeddings
        self.job_embeddings = job_embeddings

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, local_idx):
        row_idx = self.sample_indices[local_idx]
        resume_idx = self.resume_indices[row_idx]
        job_idx = self.job_indices[row_idx]
        return (
            row_idx,
            self.applicant_embeddings[resume_idx],
            self.job_embeddings[job_idx],
            self.genders[row_idx],
            self.labels[row_idx],
        )


class VectorizedLoader:
    def __init__(
        self,
        name: str,
        sample_indices: np.ndarray,
        resume_indices: np.ndarray,
        job_indices: np.ndarray,
        labels: np.ndarray,
        genders: np.ndarray,
        applicant_embeddings: np.ndarray,
        job_embeddings: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        self.name = name
        self.sample_indices = sample_indices.astype(np.int64)
        self.resume_indices = resume_indices
        self.job_indices = job_indices
        self.labels = labels
        self.genders = genders
        self.applicant_embeddings = applicant_embeddings
        self.job_embeddings = job_embeddings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.iteration = 0

    def __len__(self):
        return int(math.ceil(len(self.sample_indices) / self.batch_size))

    def __iter__(self):
        import torch

        order = self.sample_indices.copy()
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.iteration)
            rng.shuffle(order)
            self.iteration += 1
        for start in range(0, len(order), self.batch_size):
            row_idx = order[start : start + self.batch_size]
            resume_idx = self.resume_indices[row_idx]
            job_idx = self.job_indices[row_idx]
            resume_emb = np.asarray(self.applicant_embeddings[resume_idx], dtype=np.float16)
            job_emb = np.asarray(self.job_embeddings[job_idx], dtype=np.float16)
            gender = self.genders[row_idx].astype(np.float32, copy=False)
            label = self.labels[row_idx].astype(np.float32, copy=False)
            yield (
                torch.from_numpy(row_idx.astype(np.int64, copy=False)),
                torch.from_numpy(resume_emb),
                torch.from_numpy(job_emb),
                torch.from_numpy(gender),
                torch.from_numpy(label),
            )


def make_torch_components():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PairMLP(nn.Module):
        def __init__(self, emb_dim: int, proj_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.15):
            super().__init__()
            input_dim = proj_dim * 4 + 2
            self.resume_norm = nn.LayerNorm(emb_dim)
            self.job_norm = nn.LayerNorm(emb_dim)
            self.resume_proj = nn.Linear(emb_dim, proj_dim)
            self.job_proj = nn.Linear(emb_dim, proj_dim)
            self.pair_norm = nn.LayerNorm(input_dim)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(hidden_dim, 1)

        def build_features(self, resume_emb, job_emb, gender):
            resume_emb = resume_emb.float()
            job_emb = job_emb.float()
            resume_emb = F.gelu(self.resume_proj(self.resume_norm(resume_emb)))
            job_emb = F.gelu(self.job_proj(self.job_norm(job_emb)))
            diff = torch.abs(resume_emb - job_emb)
            prod = resume_emb * job_emb
            cos = F.cosine_similarity(resume_emb, job_emb, dim=1).unsqueeze(1)
            gender = (gender.float().view(-1, 1) * 2.0) - 1.0
            x = torch.cat([resume_emb, job_emb, diff, prod, cos, gender], dim=1)
            return self.pair_norm(x)

        def hidden(self, resume_emb, job_emb, gender):
            return self.encoder(self.build_features(resume_emb, job_emb, gender))

        def forward(self, resume_emb, job_emb, gender):
            return self.head(self.hidden(resume_emb, job_emb, gender)).squeeze(1)

    return PairMLP


def load_training_arrays(args):
    out_dir = Path(args.out_dir)
    applicants = pd.read_parquet(out_dir / "applicants.parquet", columns=["resume_id"])
    jobs = pd.read_parquet(out_dir / "jobs.parquet", columns=["job_id", "company_id"])
    samples = pd.read_parquet(out_dir / "samples.parquet")
    applicant_embeddings = np.load(out_dir / "applicants_embeddings.npy", mmap_mode="r")
    job_embeddings = np.load(out_dir / "jobs_embeddings.npy", mmap_mode="r")

    resume_to_idx = pd.Series(np.arange(len(applicants), dtype=np.int64), index=applicants["resume_id"]).to_dict()
    job_to_idx = pd.Series(np.arange(len(jobs), dtype=np.int64), index=jobs["job_id"]).to_dict()
    samples["resume_idx"] = samples["resume_id"].map(resume_to_idx)
    samples["job_idx"] = samples["job_id"].map(job_to_idx)
    before = len(samples)
    samples = samples.dropna(subset=["resume_idx", "job_idx"]).copy()
    if len(samples) != before:
        log(f"Dropped {before - len(samples):,} samples missing embeddings.")
    samples["resume_idx"] = samples["resume_idx"].astype(np.int64)
    samples["job_idx"] = samples["job_idx"].astype(np.int64)
    return samples, applicant_embeddings, job_embeddings


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def evaluate_model(model, loader, device, flip_gender: bool = False):
    import torch

    model.eval()
    total = 0
    correct = 0
    all_indices = []
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for row_idx, resume_emb, job_emb, gender, label in loader:
            resume_emb = resume_emb.to(device, non_blocking=True)
            job_emb = job_emb.to(device, non_blocking=True)
            gender = gender.to(device, non_blocking=True)
            if flip_gender:
                gender = 1.0 - gender
            label = label.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
                logits = model(resume_emb, job_emb, gender)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == label).sum().item()
            total += label.numel()
            all_indices.append(row_idx.cpu().numpy())
            all_probs.append(probs.float().cpu().numpy())
            all_preds.append(preds.cpu().numpy().astype(np.int8))
            all_labels.append(label.cpu().numpy().astype(np.int8))
    return {
        "accuracy": float(correct / max(total, 1)),
        "sample_indices": np.concatenate(all_indices),
        "probs": np.concatenate(all_probs),
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
    }


def fairness_metrics(samples: pd.DataFrame, base_eval: dict, cf_eval: dict) -> dict:
    order = np.argsort(base_eval["sample_indices"])
    base_indices = base_eval["sample_indices"][order]
    base_preds = base_eval["preds"][order]
    base_probs = base_eval["probs"][order]
    cf_order = np.argsort(cf_eval["sample_indices"])
    cf_indices = cf_eval["sample_indices"][cf_order]
    cf_preds = cf_eval["preds"][cf_order]
    cf_probs = cf_eval["probs"][cf_order]
    if not np.array_equal(base_indices, cf_indices):
        raise ValueError("Evaluation index mismatch between original and counterfactual runs.")

    pair_consistency = float((base_preds == cf_preds).mean())
    avg_score_gap = float(np.abs(base_probs - cf_probs).mean())
    eval_df = samples.iloc[base_indices][["resume_id", "company_id"]].copy()
    eval_df["prob"] = base_probs
    eval_df["prob_cf"] = cf_probs

    all_consistent = []
    multi_consistent = []
    for _, group in eval_df.groupby("resume_id", sort=False):
        if group.empty:
            continue
        company_a = group.iloc[group["prob"].to_numpy().argmax()]["company_id"]
        company_b = group.iloc[group["prob_cf"].to_numpy().argmax()]["company_id"]
        same = company_a == company_b
        all_consistent.append(same)
        if group["company_id"].nunique() >= 2:
            multi_consistent.append(same)
    return {
        "pair_prediction_consistency_accuracy": pair_consistency,
        "average_probability_gap": avg_score_gap,
        "top1_company_consistency_accuracy_all_resumes": float(np.mean(all_consistent)) if all_consistent else None,
        "top1_company_consistency_accuracy_multi_company_resumes": float(np.mean(multi_consistent))
        if multi_consistent
        else None,
        "top1_company_eval_resumes": int(len(all_consistent)),
        "top1_company_multi_company_eval_resumes": int(len(multi_consistent)),
    }


def train_base(args, samples, applicant_embeddings, job_embeddings):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    PairMLP = make_torch_components()

    resume_indices = samples["resume_idx"].to_numpy(np.int64)
    job_indices = samples["job_idx"].to_numpy(np.int64)
    labels = samples["label"].to_numpy(np.float32)
    genders = samples["gender"].to_numpy(np.float32)
    split = samples["split"].to_numpy()

    loaders = {}
    for name in ["train", "val", "test"]:
        sample_indices = np.where(split == name)[0]
        loaders[name] = VectorizedLoader(
            name=name,
            sample_indices=sample_indices,
            resume_indices=resume_indices,
            job_indices=job_indices,
            labels=labels,
            genders=genders,
            applicant_embeddings=applicant_embeddings,
            job_embeddings=job_embeddings,
            batch_size=args.batch_size,
            shuffle=name == "train",
            seed=args.seed,
        )
        log(f"{name} dataset: {len(sample_indices):,} samples")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emb_dim = int(applicant_embeddings.shape[1])
    model = PairMLP(emb_dim=emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    if torch.cuda.device_count() > 1 and args.use_data_parallel:
        log(f"Training with DataParallel on {torch.cuda.device_count()} GPUs, batch_size={args.batch_size}")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    best_state = None
    best_val = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        start = time.time()
        for _, resume_emb, job_emb, gender, label in loaders["train"]:
            resume_emb = resume_emb.to(device, non_blocking=True)
            job_emb = job_emb.to(device, non_blocking=True)
            gender = gender.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
                logits = model(resume_emb, job_emb, gender)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * label.numel()
            preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
            correct += (preds == label).sum().item()
            total += label.numel()
        train_acc = correct / max(total, 1)
        val_eval = evaluate_model(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": train_acc,
            "val_accuracy": val_eval["accuracy"],
            "seconds": time.time() - start,
        }
        history.append(row)
        log(f"Epoch {epoch}: train_acc={train_acc:.6f} val_acc={val_eval['accuracy']:.6f} seconds={row['seconds']:.1f}")
        if val_eval["accuracy"] > best_val:
            best_val = val_eval["accuracy"]
            state_model = model.module if hasattr(model, "module") else model
            best_state = {k: v.detach().cpu().clone() for k, v in state_model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                log(f"Early stopping after {epoch} epochs.")
                break

    state_model = model.module if hasattr(model, "module") else model
    state_model.load_state_dict(best_state)
    out_dir = Path(args.out_dir)
    torch.save({"state_dict": best_state, "history": history, "args": vars(args)}, out_dir / "base_model.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    return state_model.to(device), loaders, None, device, history, best_val


def compute_preserve_projection(model, loader, device, hidden_dim: int, max_samples: int, preserve_rank: int):
    import torch

    model.eval()
    cov = torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64, device=device)
    n = 0
    with torch.no_grad():
        for _, resume_emb, job_emb, gender, _ in loader:
            if n >= max_samples:
                break
            take = min(resume_emb.shape[0], max_samples - n)
            resume_emb = resume_emb[:take].to(device, non_blocking=True)
            job_emb = job_emb[:take].to(device, non_blocking=True)
            gender = gender[:take].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
                hidden = model.hidden(resume_emb, job_emb, gender)
            hidden = hidden.float()
            cov += hidden.T.double().matmul(hidden.double())
            n += take
    cov /= max(n, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    rank = max(1, min(preserve_rank, hidden_dim - 1))
    top = evecs[:, order[:rank]].float()
    projection = torch.eye(hidden_dim, device=device, dtype=torch.float32) - top.matmul(top.T)
    log(f"Null-space projection built from {n:,} hidden states; preserve_rank={rank}, null_dim={hidden_dim-rank}")
    return projection


def alphaedit_fairness_edit(args, base_model, loaders, samples, device, base_val_accuracy):
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    PairMLP = make_torch_components()
    emb_dim = base_model.head.in_features  # hidden dim, not embedding dim; fixed below from first layer.
    # Recreate by deep-copying the trained model through state_dict to avoid changing the base object.
    import copy

    teacher = copy.deepcopy(base_model).to(device).eval()
    student = copy.deepcopy(base_model).to(device).train()
    for p in teacher.parameters():
        p.requires_grad = False
    frozen_modules = [
        student.resume_norm,
        student.job_norm,
        student.resume_proj,
        student.job_proj,
        student.pair_norm,
        student.encoder,
    ]
    for module in frozen_modules:
        for p in module.parameters():
            p.requires_grad = False
    student.head.bias.requires_grad = False

    projection = compute_preserve_projection(
        teacher,
        loaders["train"],
        device,
        hidden_dim=args.hidden_dim,
        max_samples=args.edit_preserve_samples,
        preserve_rank=args.preserve_rank,
    )
    optimizer = optim.AdamW([student.head.weight], lr=args.edit_lr, weight_decay=0.0)

    base_val_eval = evaluate_model(teacher, loaders["val"], device)
    base_val_cf = evaluate_model(teacher, loaders["val"], device, flip_gender=True)
    base_val_fair = fairness_metrics(samples, base_val_eval, base_val_cf)
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    fairness_key = "top1_company_consistency_accuracy_multi_company_resumes"
    best_score = base_val_fair.get(fairness_key)
    if best_score is None:
        fairness_key = "pair_prediction_consistency_accuracy"
        best_score = base_val_fair[fairness_key]
    best_acc = base_val_eval["accuracy"]
    edit_history = [
        {
            "epoch": 0,
            "val_accuracy": base_val_eval["accuracy"],
            **{f"val_{k}": v for k, v in base_val_fair.items()},
        }
    ]

    for epoch in range(1, args.edit_epochs + 1):
        student.train()
        total_loss = 0.0
        total = 0
        for _, resume_emb, job_emb, gender, label in loaders["train"]:
            resume_emb = resume_emb.to(device, non_blocking=True)
            job_emb = job_emb.to(device, non_blocking=True)
            gender = gender.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            gender_cf = 1.0 - gender
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
                    target_logits = teacher(resume_emb, job_emb, gender).float()
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
                cf_logits = student(resume_emb, job_emb, gender_cf).float()
                orig_logits = student(resume_emb, job_emb, gender).float()
                fair_loss = F.mse_loss(cf_logits, target_logits)
                preserve_loss = F.mse_loss(orig_logits, target_logits)
                supervised_loss = F.binary_cross_entropy_with_logits(orig_logits, label)
                loss = fair_loss + args.preserve_loss_weight * preserve_loss + args.supervised_loss_weight * supervised_loss
            loss.backward()
            if student.head.weight.grad is not None:
                student.head.weight.grad.copy_(student.head.weight.grad.matmul(projection))
            optimizer.step()
            total_loss += loss.item() * label.numel()
            total += label.numel()

        val_eval = evaluate_model(student, loaders["val"], device)
        val_cf = evaluate_model(student, loaders["val"], device, flip_gender=True)
        val_fair = fairness_metrics(samples, val_eval, val_cf)
        row = {
            "epoch": epoch,
            "edit_loss": total_loss / max(total, 1),
            "val_accuracy": val_eval["accuracy"],
            **{f"val_{k}": v for k, v in val_fair.items()},
        }
        edit_history.append(row)
        log(
            "Edit epoch "
            f"{epoch}: val_acc={val_eval['accuracy']:.6f} "
            f"pair_fair={val_fair['pair_prediction_consistency_accuracy']:.6f} "
            f"company_fair={val_fair['top1_company_consistency_accuracy_multi_company_resumes']}"
        )
        within_tolerance = val_eval["accuracy"] >= base_val_accuracy - args.accuracy_tolerance
        score = val_fair.get(fairness_key)
        if score is None:
            score = val_fair["pair_prediction_consistency_accuracy"]
        if within_tolerance and (score > best_score or (score == best_score and val_eval["accuracy"] > best_acc)):
            best_score = score
            best_acc = val_eval["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    out_dir = Path(args.out_dir)
    torch.save({"state_dict": best_state, "history": edit_history, "args": vars(args)}, out_dir / "alphaedit_fair_model.pt")
    pd.DataFrame(edit_history).to_csv(out_dir / "fairness_edit_history.csv", index=False)
    return student.eval(), edit_history


def run_train_eval(args) -> dict:
    import torch

    samples, applicant_embeddings, job_embeddings = load_training_arrays(args)
    base_model, loaders, datasets, device, history, base_val_acc = train_base(
        args, samples, applicant_embeddings, job_embeddings
    )
    log("Evaluating base model on test split.")
    base_test = evaluate_model(base_model, loaders["test"], device)
    base_test_cf = evaluate_model(base_model, loaders["test"], device, flip_gender=True)
    base_fair = fairness_metrics(samples, base_test, base_test_cf)

    log("Running AlphaEdit-style null-space fairness edit.")
    fair_model, edit_history = alphaedit_fairness_edit(args, base_model, loaders, samples, device, base_val_acc)
    log("Evaluating edited model on test split.")
    edited_test = evaluate_model(fair_model, loaders["test"], device)
    edited_test_cf = evaluate_model(fair_model, loaders["test"], device, flip_gender=True)
    edited_fair = fairness_metrics(samples, edited_test, edited_test_cf)

    metrics = {
        "settings": {
            "embedding_model": args.embedding_model,
            "neg_ratio": args.neg_ratio,
            "split": "pair-level stratified 80/10/10 with random_state seed",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "hidden_dim": args.hidden_dim,
            "preserve_rank": args.preserve_rank,
        },
        "counts": {
            "samples": int(len(samples)),
            "train": int((samples["split"] == "train").sum()),
            "val": int((samples["split"] == "val").sum()),
            "test": int((samples["split"] == "test").sum()),
            "test_positive": int(((samples["split"] == "test") & (samples["label"] == 1)).sum()),
            "test_negative": int(((samples["split"] == "test") & (samples["label"] == 0)).sum()),
        },
        "base": {
            "recruitment_accuracy": base_test["accuracy"],
            "fairness": base_fair,
        },
        "alphaedit_nullspace": {
            "recruitment_accuracy": edited_test["accuracy"],
            "fairness": edited_fair,
        },
    }
    out_dir = Path(args.out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Saved metrics to {out_dir / 'metrics.json'}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON defaults; explicit CLI options take precedence")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--embedding-model", default="nlpai-lab/KURE-v1")
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--use-multi-gpu-embed", action="store_true", default=True)
    parser.add_argument("--no-multi-gpu-embed", dest="use_multi_gpu_embed", action="store_false")
    parser.add_argument("--fp16-embed", action="store_true", default=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--embed-chunk-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--use-data-parallel", dest="use_data_parallel", action="store_true")
    parser.add_argument("--no-data-parallel", dest="use_data_parallel", action="store_false")
    parser.set_defaults(use_data_parallel=False)
    parser.add_argument("--edit-epochs", type=int, default=3)
    parser.add_argument("--edit-lr", type=float, default=5e-3)
    parser.add_argument("--edit-preserve-samples", type=int, default=200000)
    parser.add_argument("--preserve-rank", type=int, default=384)
    parser.add_argument("--preserve-loss-weight", type=float, default=0.25)
    parser.add_argument("--supervised-loss-weight", type=float, default=0.05)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.005)
    parser.add_argument("--stage", choices=["all", "prepare", "embed", "train"], default="all")
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        config = json.loads(preliminary.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            parser.error("Config must be a JSON object.")
        known = {action.dest for action in parser._actions} - {"config", "help"}
        unknown = set(config) - known
        if unknown:
            parser.error("Unknown config keys: " + ", ".join(sorted(unknown)))
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0 or args.edit_epochs < 0:
        parser.error("Batch size and epochs must be positive; edit epochs must be nonnegative.")
    if args.config is not None:
        args.config = str(args.config)
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "run_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    if args.stage in {"all", "prepare"}:
        prepare(args)
    if args.stage in {"all", "embed"}:
        embed(args)
    if args.stage in {"all", "train"}:
        metrics = run_train_eval(args)
        log(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
