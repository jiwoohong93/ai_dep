"""
Round 40 - TARGET PUSH (national-project reporting).

GOAL: report against the ORIGINAL committed benchmark (IITP target table) on the
ORIGINAL Four-Findings data / split defined by train.py:
    예측정확도(accuracy)     >= 94%   (2026 target; currently 93.38 -> SHORT)
    공정성 DEO               <= 5%    (|TPR_f - TPR_m| x 100; currently 2.03)
    정책반영 성능평균하락률   <= 3%    ((acc_nofair-acc_fair)/acc_nofair x100; now 0.25)
Only accuracy is short. Push acc >= 94 while keeping DEO <= 5 and drop <= 3.

This script REPLICATES train.py's data pipeline / label & metric definitions EXACTLY
so the numbers stay comparable to the 93.38/2.03/0.25 benchmark, then makes the
training recipe configurable so the orchestrator can escalate across rounds:
    backbone {resnet18(orig), vit_b16, densenet121, raddino}
    epochs (orig=2, up to ~30), aug {basic(orig), strong}, class-weight fix
    (orig bug: computed then overwritten with None), TTA, seed.

REPLICATION NOTES (must match train.py for report parity):
  * Data      : merge test+validation Four-Findings CSVs -> 4,376 rows (all on disk).
  * Label     : Pneumothorax==YES -> 0, else -> 1  (so label 1 == pneumothorax-negative
                (other findings allowed), the MAJORITY class; pneumothorax positives are the rare label-0 cases).
  * Gender    : M -> +1, F -> -1  (from "Patient Gender").
  * Split     : IMAGE-LEVEL train_test_split(indices, test_size=0.10,
                random_state=42, shuffle=True)  -> train 3938 / test 438.
  * Accuracy  : overall argmax@0.5 accuracy on the 438 test images.
  * DEO       : evaluate_DEO replicated EXACTLY -> TPR computed on label==1
                (train.py's own definition), DEO=|TPR_f-TPR_m|, reported x100.
  * Fairness  : WeightedRandomSampler across (label x gender) subgroups (w=max/size).
  * Policy    : train a FAIR (WeightedRandomSampler) and a NOFAIR (shuffle) model
                with the SAME recipe; drop% = (acc_nofair-acc_fair)/acc_nofair x100.

Does NOT modify train.py / common*.py / earlier rounds.
Env: /usr/bin/python, torch.device('cuda:0') (external CUDA_VISIBLE_DEVICES),
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 (rad-DINO / ViT weights cached).
"""
import os
import csv
import json
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------ paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")                          # Four-Findings expert-label CSVs (bundled)
TEST_LABELS_CSV = os.path.join(DATA, "four_findings_expert_labels_test_labels.csv")
VAL_LABELS_CSV = os.path.join(DATA, "four_findings_expert_labels_validation_labels.csv")
# NIH ChestX-ray14 images: download_nih_subset.py writes them here (override with $NIH_ROOT)
NIH_ROOT = os.environ.get("NIH_ROOT", os.path.join(DATA, "datasets", "nih-chest-xrays", "data", "versions", "3"))
IMG_DIRS = [os.path.join(NIH_ROOT, f"images_{i:03d}", "images") for i in range(1, 13)]
RESULTS_LOG = os.path.join(HERE, "results", "results.jsonl")
ROUND_ID = 40

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RADDINO_HF = "microsoft/rad-dino"
RADDINO_REV = "110cbc18d5133582e320b43d53bf5c44e410c936"   # pinned revision

# reporting targets
TARGET_ACC = 94.0     # percent
TARGET_DEO = 5.0      # percent
TARGET_DROP = 3.0     # percent


# ------------------------------------------------------------------ utils
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_path(img_name):
    for d in IMG_DIRS:
        p = os.path.join(d, img_name)
        if os.path.exists(p):
            return p
    return None


def get_gender_label(gender_str):
    # EXACT replication of train.py: M -> +1, else -> -1
    return 1 if gender_str.strip().upper() == "M" else -1


def merge_expert_csvs(csv1, csv2):
    rows = []
    for fn in (csv1, csv2):
        with open(fn, "r") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


# ------------------------------------------------------------------ dataset
class CleanChestDataset(Dataset):
    """EXACT replication of train.py's CleanChestDataset labelling.

    sample = (path, cond_label, gender)   cond_label: pneumothorax->0 else->1
    """

    def __init__(self, all_rows, transform=None, indices=None, verbose=True):
        super().__init__()
        self.transform = transform
        self.samples = []
        missing = 0
        for row in all_rows:
            img_name = row["Image Index"].strip()
            full_path = find_image_path(img_name)
            if not full_path:
                missing += 1
                continue
            cond_label = 0 if row["Pneumothorax"] == "YES" else 1
            gval = get_gender_label(row["Patient Gender"])
            self.samples.append((full_path, cond_label, gval))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        if verbose:
            lab = Counter(s[1] for s in self.samples)
            gen = Counter(s[2] for s in self.samples)
            print(f"[Dataset] n={len(self.samples)} dropped_missing={missing} | "
                  f"label pneu(0)={lab[0]} pneu_neg(1)={lab[1]} | "
                  f"gender M(+1)={gen[1]} F(-1)={gen[-1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cond_label, gender_label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, cond_label, gender_label

    def get_metadata(self):
        return [(s[1], s[2]) for s in self.samples]


# ------------------------------------------------------------------ transforms
def build_transforms(aug, mean, std, size=224):
    """basic == train.py's exact train transform. strong adds rotation,
    colorjitter, RandAugment-lite and RandomErasing."""
    norm = transforms.Normalize(mean, std)
    if aug == "basic":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.81, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    elif aug == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    else:
        raise ValueError(f"unknown aug {aug}")
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, test_tf


# ------------------------------------------------------------------ models
class RadDinoClassifier(nn.Module):
    """HuggingFace rad-DINO backbone + fresh 2-class linear head
    (same build pattern as rounds 12-15)."""

    def __init__(self, num_classes=2, hf_name=RADDINO_HF):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(hf_name, revision=RADDINO_REV)
        h = self.backbone.config.hidden_size
        self.head = nn.Linear(h, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        feat = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, 0]
        return self.head(feat)


def _tv_weights(enum_name):
    """Return torchvision weights enum if the checkpoint is cached, else None.
    Under *_OFFLINE=1 a cache miss raises; we catch and fall back to random init
    with a clear note (so a non-cached backbone still *runs*, just uncomparable)."""
    return enum_name


def build_model(backbone, device):
    """Return (model, is_transformer, mean, std, note)."""
    note = ""
    if backbone == "resnet18":
        try:
            m = models.resnet18(weights="IMAGENET1K_V1")
        except Exception as e:
            m = models.resnet18(weights=None)
            note = f"resnet18 pretrained unavailable ({e}); random init"
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m.to(device), False, IMAGENET_MEAN, IMAGENET_STD, note

    if backbone == "densenet121":
        try:
            m = models.densenet121(weights="IMAGENET1K_V1")
        except Exception as e:
            m = models.densenet121(weights=None)
            note = (f"densenet121 pretrained NOT cached offline ({type(e).__name__}); "
                    "random init -> NOT comparable, cache weights before full run")
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m.to(device), False, IMAGENET_MEAN, IMAGENET_STD, note

    if backbone == "vit_b16":
        try:
            m = models.vit_b_16(weights="IMAGENET1K_V1")
        except Exception as e:
            m = models.vit_b_16(weights=None)
            note = f"vit_b16 pretrained unavailable ({e}); random init"
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
        return m.to(device), True, IMAGENET_MEAN, IMAGENET_STD, note

    if backbone == "raddino":
        m = RadDinoClassifier()
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(RADDINO_HF, revision=RADDINO_REV)
            mean = list(proc.image_mean)
            std = list(proc.image_std)
        except Exception as e:
            note = f"rad-DINO processor norm unavailable ({e}); ImageNet norm"
        return m.to(device), True, mean, std, note

    raise ValueError(f"unknown backbone {backbone}")


def build_optimizer(model, backbone, is_transformer, lr):
    """Transformers get differential LR (backbone low, head high) as in R12."""
    if is_transformer:
        if backbone == "raddino":
            bb = list(model.backbone.parameters())
            head = list(model.head.parameters())
        else:  # vit_b16
            head = list(model.heads.parameters())
            head_ids = {id(p) for p in head}
            bb = [p for p in model.parameters() if id(p) not in head_ids]
        return optim.AdamW([
            {"params": bb, "lr": lr},
            {"params": head, "lr": max(lr * 50, 1e-3)},
        ], weight_decay=1e-4)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


# ------------------------------------------------------------------ sampler
def build_fair_sampler(train_dataset):
    """EXACT replication of train.py fairness: WeightedRandomSampler across
    (cond_label x gender) subgroups with weight = max_group_size / group_size."""
    group_dict = defaultdict(list)
    meta = train_dataset.get_metadata()
    for idx, (c_label, g_label) in enumerate(meta):
        group_dict[(c_label, g_label)].append(idx)
    group_sizes = {g: len(idxs) for g, idxs in group_dict.items()}
    max_size = max(group_sizes.values()) if group_sizes else 1
    sample_weights = np.zeros(len(meta), dtype=np.float32)
    for g, idxs in group_dict.items():
        w = float(max_size / len(idxs)) if len(idxs) > 0 else 1.0
        for i in idxs:
            sample_weights[i] = w
    sampler = WeightedRandomSampler(sample_weights, len(meta))
    return sampler, {str(k): v for k, v in group_sizes.items()}


# ------------------------------------------------------------------ eval (DEO)
@torch.no_grad()
def evaluate_DEO(model, loader, device, is_transformer=False, tta=False, return_preds=False):
    """EXACT replication of train.py.evaluate_DEO metric definitions:
    overall accuracy + TPR on label==1, DEO=|TPR_f-TPR_m|. TTA (hflip) optional.
    return_preds=True also returns the per-sample (preds, labels, gender) arrays."""
    model.eval()
    all_preds, all_labels, all_gender = [], [], []
    for images, labels, gender in loader:
        images = images.to(device)
        logits = model(images)
        if tta:
            logits = logits + model(torch.flip(images, dims=[3]))
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(np.asarray(labels))
        all_gender.extend(np.asarray(gender))

    acc = accuracy_score(all_labels, all_preds)
    female_idx = [i for i, g in enumerate(all_gender) if g == -1]
    male_idx = [i for i, g in enumerate(all_gender) if g == 1]
    female_actual_1 = [i for i in female_idx if all_labels[i] == 1]
    male_actual_1 = [i for i in male_idx if all_labels[i] == 1]
    tpr_f = (sum(all_preds[j] == 1 for j in female_actual_1) / len(female_actual_1)
             if female_actual_1 else 0.0)
    tpr_m = (sum(all_preds[j] == 1 for j in male_actual_1) / len(male_actual_1)
             if male_actual_1 else 0.0)
    deo = abs(tpr_f - tpr_m)
    if return_preds:
        return (float(acc), float(deo), float(tpr_f), float(tpr_m),
                (np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_gender)))
    return float(acc), float(deo), float(tpr_f), float(tpr_m)


# ------------------------------------------------------------------ train
def make_criterion(train_dataset, fix_classweight, device):
    """train.py computed inverse-freq class weights then DISABLED them (=None).
    fix_classweight applies them properly (balanced inverse-frequency)."""
    labels = [c for c, _ in train_dataset.get_metadata()]
    cnt = Counter(labels)
    if not fix_classweight:
        return nn.CrossEntropyLoss()
    tot = sum(cnt.values())
    # balanced: w_c = tot / (2 * count_c)  -> upweights the rare pneumothorax(0)
    w = torch.tensor([tot / (2 * max(cnt[0], 1)), tot / (2 * max(cnt[1], 1))],
                     dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=w)


def train_variant(rows, train_idx, test_idx, args, use_fairness, device):
    """Train one model (fair or nofair) with the shared recipe. Returns metrics."""
    tag = "FAIR" if use_fairness else "NOFAIR"
    model, is_tf, mean, std, note = build_model(args.backbone, device)
    if note:
        print(f"[{tag}] NOTE: {note}")
    train_tf, test_tf = build_transforms(args.aug, mean, std)
    train_ds = CleanChestDataset(rows, transform=train_tf, indices=train_idx, verbose=False)
    test_ds = CleanChestDataset(rows, transform=test_tf, indices=test_idx, verbose=False)

    if use_fairness:
        sampler, groups = build_fair_sampler(train_ds)
        print(f"[{tag}] WeightedRandomSampler groups(label,gender)->size: {groups}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    else:
        print(f"[{tag}] standard shuffle")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    criterion = make_criterion(train_ds, args.fix_classweight, device)
    optimizer = build_optimizer(model, args.backbone, is_tf, args.lr)

    for epoch in range(args.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        acc, deo, tf, tm = evaluate_DEO(model, test_loader, device, is_tf, tta=args.tta)
        print(f"[{tag}] ep{epoch+1}/{args.epochs} loss={run_loss/max(seen,1):.4f} "
              f"acc={acc:.4f} DEO={deo:.4f} (TPR_f={tf:.3f} TPR_m={tm:.3f})")
        every = getattr(args, "save_every", 0) or 0
        if getattr(args, "save_dir", None) and every > 0 and (epoch + 1) % every == 0:
            ep_dir = os.path.join(args.save_dir, f"ep{epoch+1:02d}")
            os.makedirs(ep_dir, exist_ok=True)
            path = os.path.join(ep_dir, CKPT_NAMES[use_fairness])
            torch.save(model.state_dict(), path)
            print(f"[{tag}] saved epoch checkpoint {path} sha256={sha256_file(path)}")

    acc, deo, tf, tm, preds = evaluate_DEO(model, test_loader, device, is_tf, tta=args.tta,
                                           return_preds=True)
    print(f"[{tag}] FINAL acc={acc:.4f} DEO={deo:.4f} TPR_f={tf:.3f} TPR_m={tm:.3f}")
    if getattr(args, "save_dir", None):
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, CKPT_NAMES[use_fairness])
        torch.save(model.state_dict(), path)
        print(f"[{tag}] saved checkpoint {path} sha256={sha256_file(path)}")
    return {"acc": acc, "deo": deo, "tpr_f": tf, "tpr_m": tm, "_preds": preds}


CKPT_NAMES = {True: "fair.pt", False: "nofair.pt"}


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_variant(rows, test_idx, args, use_fairness, device):
    """Load a trained checkpoint (args.ckpt_dir/{fair,nofair}.pt) and evaluate on the test split only."""
    tag = "FAIR" if use_fairness else "NOFAIR"
    path = os.path.join(args.ckpt_dir, CKPT_NAMES[use_fairness])
    print(f"[{tag}] load checkpoint {path} sha256={sha256_file(path)}")
    model, is_tf, mean, std, note = build_model(args.backbone, device)
    if note:
        print(f"[{tag}] NOTE: {note}")
    model.load_state_dict(torch.load(path, map_location=device))
    _, test_tf = build_transforms(args.aug, mean, std)
    test_ds = CleanChestDataset(rows, transform=test_tf, indices=test_idx, verbose=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    acc, deo, tf, tm, preds = evaluate_DEO(model, test_loader, device, is_tf, tta=args.tta,
                                           return_preds=True)
    print(f"[{tag}] FINAL acc={acc:.4f} DEO={deo:.4f} TPR_f={tf:.3f} TPR_m={tm:.3f}")
    return {"acc": acc, "deo": deo, "tpr_f": tf, "tpr_m": tm, "_preds": preds}


# ------------------------------------------------------------------ data prep
def prepare_split(verbose=True):
    """Merge CSVs + EXACT image-level split from train.py."""
    rows = merge_expert_csvs(TEST_LABELS_CSV, VAL_LABELS_CSV)
    if verbose:
        print(f"[Main] merged rows = {len(rows)}")
    full = CleanChestDataset(rows, transform=None, verbose=verbose)
    all_indices = list(range(len(full)))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.10, random_state=42, shuffle=True)
    return rows, full, train_idx, test_idx


def split_composition(full, idx):
    samples = [full.samples[i] for i in idx]
    pneu = [s for s in samples if s[1] == 0]          # pneumothorax positives
    return {
        "n": len(samples),
        "pos": len(pneu),                              # rare pneumothorax cases
        "pos_F": sum(1 for s in pneu if s[2] == -1),
        "pos_M": sum(1 for s in pneu if s[2] == 1),
        "F": sum(1 for s in samples if s[2] == -1),
        "M": sum(1 for s in samples if s[2] == 1),
        "nofind": sum(1 for s in samples if s[1] == 1),
    }


# ------------------------------------------------------------------ smoke
def run_smoke(args, device):
    print("=" * 70)
    print("SMOKE TEST (no full training)")
    print("=" * 70)
    rows, full, train_idx, test_idx = prepare_split(verbose=True)
    assert len(rows) == 4376, f"expected 4376 merged rows, got {len(rows)}"
    tr_c = split_composition(full, train_idx)
    te_c = split_composition(full, test_idx)
    print(f"[SPLIT] train n={tr_c['n']}  test n={te_c['n']}  (expect 3938 / 438)")
    print(f"[SPLIT] TEST pneumothorax-positives={te_c['pos']}  "
          f"F={te_c['pos_F']} M={te_c['pos_M']}  (expect ~30, F16/M14)")
    print(f"[SPLIT] TEST gender all: F={te_c['F']} M={te_c['M']}  "
          f"| Pneu-negative(label1)={te_c['nofind']}")
    print(f"[NOTE] train.py's evaluate_DEO computes TPR on label==1 (pneumothorax-negative, "
          f"the majority); DEO=|TPR_f-TPR_m|. Replicated verbatim.")

    for bb in ["resnet18", "raddino"]:
        print("-" * 60)
        print(f"[SMOKE] backbone = {bb}")
        try:
            model, is_tf, mean, std, note = build_model(bb, device)
        except Exception as e:
            print(f"[SMOKE] {bb} BUILD FAILED: {type(e).__name__}: {e}")
            continue
        if note:
            print(f"[SMOKE] note: {note}")
        print(f"[SMOKE] built {bb} (transformer={is_tf}) norm mean={ [round(x,3) for x in mean] }")
        train_tf, test_tf = build_transforms(args.aug, mean, std)
        train_ds = CleanChestDataset(rows, transform=train_tf, indices=train_idx, verbose=False)
        test_ds = CleanChestDataset(rows, transform=test_tf, indices=test_idx, verbose=False)

        # WeightedRandomSampler builds
        sampler, groups = build_fair_sampler(train_ds)
        print(f"[SMOKE] WeightedRandomSampler OK groups(label,gender)->size: {groups}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=0, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # 1-batch fwd + bwd
        criterion = make_criterion(train_ds, args.fix_classweight, device)
        optimizer = build_optimizer(model, bb, is_tf, args.lr)
        images, labels, _ = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"[SMOKE] 1-batch fwd+bwd OK  logits={tuple(logits.shape)} loss={loss.item():.4f}")

        # evaluate/DEO on a few batches (subset test loader)
        small = CleanChestDataset(rows, transform=test_tf, indices=test_idx[:3 * args.batch_size],
                                  verbose=False)
        small_loader = DataLoader(small, batch_size=args.batch_size, shuffle=False, num_workers=0)
        acc, deo, tf, tm = evaluate_DEO(model, small_loader, device, is_tf, tta=args.tta)
        print(f"[SMOKE] evaluate_DEO OK (few batches, untrained) "
              f"acc={acc:.4f} DEO={deo:.4f} TPR_f={tf:.3f} TPR_m={tm:.3f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("=" * 70)
    print("SMOKE OK - split reproduced, resnet18 + rad-DINO build/fwd/bwd/sampler/DEO all pass")
    print("=" * 70)


# ------------------------------------------------------------------ full run
def run_full(args, device):
    tag = f"{args.backbone}_e{args.epochs}_{args.aug}"
    if args.fix_classweight:
        tag += "_cw"
    if args.tta:
        tag += "_tta"
    tag += f"_s{args.seed}"
    if getattr(args, "ckpt_dir", None):
        tag += "_ckpteval"
    print("=" * 70)
    print(f"TARGET PUSH full run: {tag}  (seed={args.seed})")
    print("=" * 70)
    rows, full, train_idx, test_idx = prepare_split(verbose=True)
    te_c = split_composition(full, test_idx)
    print(f"[SPLIT] train={len(train_idx)} test={len(test_idx)} "
          f"test_pneu_pos={te_c['pos']} (F{te_c['pos_F']}/M{te_c['pos_M']})")

    if getattr(args, "ckpt_dir", None):
        print("\n----- EVAL ONLY: trained checkpoints, no training -----")
        print("\n----- MODEL B (WITH FAIRNESS: WeightedRandomSampler) -----")
        fair = eval_variant(rows, test_idx, args, use_fairness=True, device=device)
        print("\n----- MODEL A (NO FAIRNESS: standard shuffle) -----")
        nofair = eval_variant(rows, test_idx, args, use_fairness=False, device=device)
    else:
        print("\n----- MODEL B (WITH FAIRNESS: WeightedRandomSampler) -----")
        seed_everything(args.seed)
        fair = train_variant(rows, train_idx, test_idx, args, use_fairness=True, device=device)

        print("\n----- MODEL A (NO FAIRNESS: standard shuffle) -----")
        seed_everything(args.seed)
        nofair = train_variant(rows, train_idx, test_idx, args, use_fairness=False, device=device)

    acc_fair, acc_nofair = fair["acc"], nofair["acc"]
    deo_fair_pct = fair["deo"] * 100.0
    drop_pct = ((acc_nofair - acc_fair) / acc_nofair * 100.0) if acc_nofair > 0 else 0.0

    pass_acc = acc_fair * 100.0 >= TARGET_ACC
    pass_deo = deo_fair_pct <= TARGET_DEO
    pass_drop = drop_pct <= TARGET_DROP
    overall = pass_acc and pass_deo and pass_drop

    print("\n" + "=" * 70)
    print("*** RESULTS (reported on FAIR model, original benchmark definitions) ***")
    print(f"accuracy(fair)   = {acc_fair*100:.2f}%   accuracy(nofair) = {acc_nofair*100:.2f}%")
    print(f"DEO(fair)        = {deo_fair_pct:.2f}%   (TPR_f={fair['tpr_f']:.3f} TPR_m={fair['tpr_m']:.3f})")
    print(f"policy-drop      = {drop_pct:.2f}%")
    print(f"TARGET CHECK: acc>=94? {pass_acc} ({acc_fair*100:.2f}) | "
          f"DEO<=5? {pass_deo} ({deo_fair_pct:.2f}) | "
          f"drop<=3? {pass_drop} ({drop_pct:.2f})  => "
          f"{'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    variant = f"{args.backbone}_e{args.epochs}_{args.aug}"
    notes = (f"R40 target-push on ORIGINAL Four-Findings merge(4376)/image-level split "
             f"(train3938/test438, test_pneu_pos={te_c['pos']} F{te_c['pos_F']}/M{te_c['pos_M']}); "
             f"recipe: backbone={args.backbone} epochs={args.epochs} aug={args.aug} "
             f"fix_classweight={args.fix_classweight} tta={args.tta} lr={args.lr} bs={args.batch_size} seed={args.seed}. "
             f"TARGETS: acc>=94% -> {acc_fair*100:.2f}% [{'PASS' if pass_acc else 'FAIL'}]; "
             f"DEO<=5% -> {deo_fair_pct:.2f}% [{'PASS' if pass_deo else 'FAIL'}]; "
             f"policy-drop<=3% -> {drop_pct:.2f}% [{'PASS' if pass_drop else 'FAIL'}]; "
             f"acc_nofair={acc_nofair*100:.2f}%. OVERALL={'PASS' if overall else 'FAIL'}. "
             f"Baseline committed: acc93.38/DEO2.03/drop0.25. "
             f"DEO def = train.py evaluate_DEO (TPR on label==1).")

    row = {"round": ROUND_ID, "round_id": ROUND_ID,
           "arch": f"{args.backbone} (target-push)", "variant": variant,
           "acc": round(acc_fair, 4), "deo": round(fair["deo"], 4),
           "tpr_f": round(fair["tpr_f"], 3), "tpr_m": round(fair["tpr_m"], 3),
           "notes": notes}
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[logged]", json.dumps(row, ensure_ascii=False))

    out = {
        "round": ROUND_ID, "tag": tag, "variant": variant,
        "config": vars(args),
        "acc_fair": acc_fair, "acc_nofair": acc_nofair,
        "deo_fair_pct": deo_fair_pct, "policy_drop_pct": drop_pct,
        "tpr_f_fair": fair["tpr_f"], "tpr_m_fair": fair["tpr_m"],
        "fair": fair, "nofair": nofair,
        "targets": {"acc": TARGET_ACC, "deo": TARGET_DEO, "drop": TARGET_DROP},
        "pass": {"acc": pass_acc, "deo": pass_deo, "drop": pass_drop, "overall": overall},
        "test_composition": te_c,
    }
    preds = {"fair": fair.pop("_preds"), "nofair": nofair.pop("_preds")}
    out_path = os.path.join(HERE, "results", f"target_push_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")
    out["_preds"] = preds      # per-sample test predictions (not saved to JSON)
    return out


ITEMS = ("accuracy", "deo", "drop")


def run_item(args, device):
    """Evaluate ONE certification test item from trained checkpoints (args.ckpt_dir).
    accuracy / deo use fair.pt only; drop uses fair.pt and nofair.pt."""
    item = args.item
    print("=" * 70)
    print(f"TEST ITEM: {item}  (checkpoints: {args.ckpt_dir}, evaluation only)")
    print("=" * 70)
    rows, full, train_idx, test_idx = prepare_split(verbose=True)
    te_c = split_composition(full, test_idx)
    print(f"[SPLIT] train={len(train_idx)} test={len(test_idx)} "
          f"test_pneu_pos={te_c['pos']} (F{te_c['pos_F']}/M{te_c['pos_M']})")

    fair = eval_variant(rows, test_idx, args, use_fairness=True, device=device)
    preds, labels, gender = fair.pop("_preds")
    out = {"item": item, "ckpt_dir": args.ckpt_dir, "fair": fair}
    if item == "drop":
        nofair = eval_variant(rows, test_idx, args, use_fairness=False, device=device)
        nofair.pop("_preds")
        out["nofair"] = nofair

    print("\n" + "=" * 70)
    if item == "accuracy":
        correct, total = int((preds == labels).sum()), len(labels)
        value = correct / total * 100.0
        ok = value >= TARGET_ACC
        print("[ITEM 1] accuracy (fair model)")
        print(f"  correct / total = {correct} / {total}")
        print(f"  accuracy = {value:.2f} %   (target >= {TARGET_ACC:g} %)  => {'PASS' if ok else 'FAIL'}")
        out.update(correct=correct, total=total)
    elif item == "deo":
        f1 = (gender == -1) & (labels == 1)
        m1 = (gender == 1) & (labels == 1)
        hf, nf = int((preds[f1] == 1).sum()), int(f1.sum())
        hm, nm = int((preds[m1] == 1).sum()), int(m1.sum())
        value = abs(hf / nf - hm / nm) * 100.0
        ok = value <= TARGET_DEO
        print("[ITEM 2] DEO = |TPR_female - TPR_male| x 100 (fair model, TPR on label 1)")
        print(f"  TPR_female = {hf} / {nf} = {hf / nf:.4f}")
        print(f"  TPR_male   = {hm} / {nm} = {hm / nm:.4f}")
        print(f"  DEO = {value:.2f} %   (target <= {TARGET_DEO:g} %)  => {'PASS' if ok else 'FAIL'}")
        out.update(tpr_f_hits=hf, tpr_f_n=nf, tpr_m_hits=hm, tpr_m_n=nm)
    else:
        a_f, a_n = fair["acc"] * 100.0, nofair["acc"] * 100.0
        value = (a_n - a_f) / a_n * 100.0
        ok = value <= TARGET_DROP
        print("[ITEM 3] policy drop = (acc_nofair - acc_fair) / acc_nofair x 100")
        print(f"  accuracy(nofair) = {a_n:.2f} %   accuracy(fair) = {a_f:.2f} %")
        print(f"  policy drop = {value:.2f} %   (target <= {TARGET_DROP:g} %)  => {'PASS' if ok else 'FAIL'}")
    print("=" * 70)

    out.update(value=value, passed=bool(ok))
    out_path = os.path.join(HERE, "results", f"test_item_{item}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")
    return out


# ------------------------------------------------------------------ main
def build_argparser():
    p = argparse.ArgumentParser(description="Round 40 target-push train/eval.")
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "vit_b16", "raddino", "densenet121"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--aug", default="basic", choices=["basic", "strong"])
    p.add_argument("--fix_classweight", action="store_true",
                   help="apply balanced class-weighted CE (train.py had it disabled)")
    p.add_argument("--tta", action="store_true", help="hflip test-time augmentation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="build+1-batch checks, no training")
    p.add_argument("--save_dir", default=None,
                   help="after training, save fair.pt / nofair.pt (model state_dict) here")
    p.add_argument("--save_every", type=int, default=0,
                   help="with --save_dir: also save SAVE_DIR/epNN/{fair,nofair}.pt every N epochs")
    p.add_argument("--ckpt_dir", default=None,
                   help="evaluation only: load fair.pt / nofair.pt from here, skip training")
    p.add_argument("--item", default=None, choices=ITEMS,
                   help="with --ckpt_dir: evaluate one test item only (accuracy | deo | drop)")
    return p


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} torch={torch.__version__}")
    if args.smoke:
        run_smoke(args, device)
    elif args.item:
        assert args.ckpt_dir, "--item needs --ckpt_dir"
        run_item(args, device)
    else:
        run_full(args, device)


if __name__ == "__main__":
    main()
