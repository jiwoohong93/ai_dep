"""
Subgroup-Balanced Resampling  (pre-processing)
그룹 균형 리샘플링 — MAF contribution, ALINMed.

Chest X-ray pneumothorax classification on the Four-Findings expert-label benchmark
(NIH ChestX-ray14 images, 4,376 images, image-level split random_state=42 ->
train 3,938 / test 438), exactly as in performance_targets/target_push.py:

  - BASELINE (metrics_orig) : model trained with standard shuffling
  - ②        (metrics_transf): same model trained with a WeightedRandomSampler over
                               (label x gender) groups, weight = max_group_size / group_size

Training/evaluation code is performance_targets/target_push.py, unchanged. The test
predictions are converted to AIF360 `aifData` and scored with metrics.common_utils
(MAF2024_guide format); the three project targets (accuracy, DEO, policy drop) are
printed as well.

Label convention of the original project (train.py): 0 = Pneumothorax, 1 = pneumothorax-negative (other findings allowed);
DEO = |TPR_F - TPR_M| on label 1. favorable label = 1, protected = sex (1 = M, 0 = F).

Run
---
  python algorithms/preprocessing/SubgroupBalancedResampling.py                       # rad-DINO, seed 42
  python algorithms/preprocessing/SubgroupBalancedResampling.py --seed 1              # other seed
  python algorithms/preprocessing/SubgroupBalancedResampling.py --backbone resnet18   # ResNet-18 baseline
"""
import os
import sys
import argparse

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'performance_targets'))

import numpy as np
import pandas as pd
import torch
from aif360.datasets import StandardDataset

from metrics import common_utils
import target_push

# recipes used for the reported results (performance_targets/README.md)
RECIPES = {
    'raddino':  dict(epochs=15, aug='strong', fix_classweight=True, tta=True,  lr=2e-5, batch_size=16),
    'resnet18': dict(epochs=20, aug='strong', fix_classweight=True, tta=False, lr=5e-5, batch_size=16),
}


class aifData(StandardDataset):
    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None):
        super(aifData, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)


class Subgroup_Balanced_Resampling:
    def __init__(self, backbone: str = 'raddino', seed: int = 42,
                 ckpt_dir: str = None, save_dir: str = None, save_every: int = 0) -> None:
        self.backbone = backbone
        self.seed = seed
        self.args = target_push.build_argparser().parse_args([])
        self.args.backbone = backbone
        self.args.seed = seed
        self.args.ckpt_dir = ckpt_dir      # evaluate trained checkpoints only (no training)
        self.args.save_dir = save_dir      # save checkpoints after training
        self.args.save_every = save_every  # also save save_dir/epNN/ every N epochs
        for k, v in RECIPES[backbone].items():
            setattr(self.args, k, v)
        self.privileged_groups = [{'sex': 1}]
        self.unprivileged_groups = [{'sex': 0}]

    def to_aif(self, preds, labels, gender):
        """Test predictions -> (dataset_true, dataset_pred) as aifData."""
        df = pd.DataFrame({'Finding': np.asarray(labels, int),
                           'sex': (np.asarray(gender) == 1).astype(int)})
        dataset_true = aifData(df=df, label_name='Finding', favorable_classes=[1],
                               protected_attribute_names=['sex'], privileged_classes=[[1]])
        dataset_pred = dataset_true.copy(deepcopy=True)
        dataset_pred.labels = np.asarray(preds, float).reshape(-1, 1)
        return dataset_true, dataset_pred

    def run(self):
        target_push.seed_everything(self.seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        out = target_push.run_full(self.args, device)
        self.kpi = {k: out[k] for k in ('acc_fair', 'acc_nofair', 'deo_fair_pct', 'policy_drop_pct')}

        metrics = []
        for variant in ('nofair', 'fair'):          # BASELINE, then ②
            preds, labels, gender = out['_preds'][variant]
            dataset_true, dataset_pred = self.to_aif(preds, labels, gender)
            metrics.append(common_utils.compute_metrics(
                dataset_true, dataset_pred,
                self.unprivileged_groups, self.privileged_groups, disp=False))
        metrics_orig, metrics_transf = metrics
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='raddino', choices=list(RECIPES))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_dir', default=None,
                    help='evaluate trained checkpoints (fair.pt, nofair.pt) only, no training')
    ap.add_argument('--save_dir', default=None, help='save fair.pt / nofair.pt after training')
    ap.add_argument('--save_every', type=int, default=0,
                    help='with --save_dir: also save SAVE_DIR/epNN/{fair,nofair}.pt every N epochs')
    ap.add_argument('--item', default=None, choices=target_push.ITEMS,
                    help='with --ckpt_dir: run one test item only (accuracy | deo | drop)')
    a = ap.parse_args()

    clf = Subgroup_Balanced_Resampling(backbone=a.backbone, seed=a.seed,
                                       ckpt_dir=a.ckpt_dir, save_dir=a.save_dir,
                                       save_every=a.save_every)
    if a.item:                                   # 공인시험: 항목별 단독 실행
        if not a.ckpt_dir:
            ap.error('--item needs --ckpt_dir')
        clf.args.item = a.item
        target_push.seed_everything(a.seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        target_push.run_item(clf.args, device)
        sys.exit(0)
    m_orig, m_transf = clf.run()
    keys = ["accuracy", "Balanced accuracy", "true_positive_rate",
            "equal_opportunity_difference", "average_abs_odds_difference"]
    print(f"\n{'metric':32s}{'BEFORE(baseline)':>18s}{'AFTER(②)':>18s}")
    for k in keys:
        print(f"{k:32s}{m_orig[k]:>18.4f}{m_transf[k]:>18.4f}")
    print(f"\n[targets] accuracy {clf.kpi['acc_fair']*100:.2f}% (>=94) | "
          f"DEO {clf.kpi['deo_fair_pct']:.2f}% (<=5) | policy drop {clf.kpi['policy_drop_pct']:.2f}% (<=3)")
