"""ai_dep financial_credit_1 (FairLogitNet) 재학습 + DEO 재측정.

원 리포의 사전학습 체크포인트(Google Drive)가 404라 train_fairness.py와 동일 설정
(seed 210, Adagrad lr 0.01, batch 256, 50 epoch, 반사실 앙상블 w=1.52/2.94)으로
재학습하고, 매 에폭 test에서 세 지표를 잰다:

- acc        : 전체 정확도 (리포 compute_accuracy와 동일 의미)
- deo_repo   : 리포 구현 그대로 — 그룹별 "전체 정확도" 격차 (labels==1 필터 없음)
- deo_lit    : 문헌 정의 — y==1 조건 TPR 격차 (Hardt et al. 2016)

원 리포가 체크포인트를 고른 기준(acc>0.81 & deo_repo<0.07, test 기준 선택)도 그대로
기록한다. 실행: Regulation과 무관, 어디서든
    python Bias/src/eval_ai_dep_deo.py
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

EXTERNAL = Path(__file__).resolve().parents[1] / "external/ai_dep/financial_credit/financial_credit_1"
sys.path.insert(0, str(EXTERNAL))
os.chdir(EXTERNAL)  # dataloader가 data/datasets/... 상대경로를 씀

from fairness.dataloader import loadDataset
from fairness.model import LearnerNN
from fairness.metrics_fairness import compute_accuracy, compute_deo

OUT_DIR = Path(__file__).resolve().parents[1] / "result/ai_dep_deo"


def set_global_seed(seed):
    np.random.seed(seed % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def ensemble_predictions(model, loader, device):
    """리포와 동일한 반사실 앙상블 예측 + (sex, y)를 평탄화해 반환."""
    preds, sexes, ys = [], [], []
    for feat_cat, feat_num, labels in loader:
        feat_cat, feat_num = feat_cat.to(device), feat_num.to(device)
        logit, _, _ = model(feat_cat, feat_num)
        c_logit, _, _ = model(1 - feat_cat, feat_num)
        w = torch.where(feat_cat == 1, 1.52, 2.94)
        cw = torch.where(1 - feat_cat == 1, 1.52, 2.94)
        p = torch.sigmoid(logit * w + c_logit * cw).round()
        preds.append(p.flatten().cpu())
        sexes.append(feat_cat.flatten().cpu())
        ys.append(labels.flatten().cpu())
    return torch.cat(preds), torch.cat(sexes), torch.cat(ys)


def deo_literature(preds, sex, y):
    """DEO_lit = |P(ŷ=1|y=1,남) − P(ŷ=1|y=1,여)| (남=1, 여=0)."""
    tpr_m = preds[(sex == 1) & (y == 1)].mean().item()
    tpr_f = preds[(sex == 0) & (y == 1)].mean().item()
    return abs(tpr_m - tpr_f), tpr_m, tpr_f


def main():
    set_global_seed(210)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = loadDataset(dataset="uci_adult", train_or_test="train", embedding_size=32)
    test_ds = loadDataset(dataset="uci_adult", train_or_test="test")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = LearnerNN(
        train_ds.categorical_embedding_sizes,
        len(train_ds.mean_std.keys()),
        [64, 32],
        activation_fn=nn.ReLU,
        device=device,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(1, 51):
        model.train()
        for cat, num, target in train_loader:
            optimizer.zero_grad()
            cat, num, target = cat.to(device), num.to(device), target.to(device)
            logit, _, _ = model(cat, num)
            c_logit, _, _ = model(1 - cat, num)
            w = torch.where(cat == 1, 1.52, 2.94)
            cw = torch.where(1 - cat == 1, 1.52, 2.94)
            loss = criterion(logit * w + c_logit * cw, target).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        acc = compute_accuracy(model, test_loader, device)
        deo_repo = compute_deo(model, test_loader, device)
        preds, sex, y = ensemble_predictions(model, test_loader, device)
        deo_lit, tpr_m, tpr_f = deo_literature(preds, sex, y)
        row = {"epoch": epoch, "acc": round(acc, 4), "deo_repo": round(deo_repo, 4),
               "deo_lit": round(deo_lit, 4), "tpr_male": round(tpr_m, 4),
               "tpr_female": round(tpr_f, 4)}
        history.append(row)
        print(row, flush=True)
        if acc > 0.81 and deo_repo < 0.07 and best is None:  # 원 리포의 저장 기준(첫 도달)
            best = dict(row)
            torch.save(model.state_dict(), OUT_DIR / "fairness_model_repro.pth")

    with (OUT_DIR / "history.json").open("w") as f:
        json.dump({"repo_criterion_first_hit": best, "history": history}, f, indent=2)
    print(f"\nsaved: {OUT_DIR}/history.json")
    print("repo criterion (acc>0.81 & deo_repo<0.07) first hit:", best)


if __name__ == "__main__":
    main()
