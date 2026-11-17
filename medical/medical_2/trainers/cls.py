import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from trainers.base import BaseTrainer
from utils.metrics import evaluate_binary, organize_results


class CLSTrainer(BaseTrainer):
    """Glaucoma classifier trainer.

    --method erm  : baseline. Class-weighted cross entropy over the whole batch.
    --method fair : FairFundusNet. Batches come from FairBatchSampler (half of each sensitive
                    group) and the loss is reweighted per group: each group's loss is weighted
                    by the *other* group's share of the training set, so the minority group
                    contributes more.
    """

    def __init__(self, args, model, train_data, test_loader, logger) -> None:
        super().__init__(args, model, logger)
        self.test_loader = test_loader

        self.class_weights = train_data.class_weights_y.to(self.device) if args.cls_balance else None
        # share of each sensitive group in the training set, e.g. [0.6923, 0.3077] for Age
        self.group_proportions = train_data.get_A_proportions()
        # fair: each group's loss is weighted by the other group's share unless given explicitly
        self.group_loss_weights = args.group_loss_weights or [1 - p for p in self.group_proportions]

        self.best_key = None
        self.eval_probs = []

    def selection_key(self, summary):
        """Ranking of a test evaluation for picking the best checkpoint (higher is better).

        Both methods: higher accuracy first, then smaller accuracy gap between groups.
        fair additionally prefers checkpoints that reach the accuracy and gap targets.
        NOTE: this selects the checkpoint on the test set (there is no validation split).
        """
        acc, gap = summary["acc@best_f1"], summary["acc_gap@best_f1"]
        key = (round(acc, 6), -round(gap, 6))
        if self.args.method == "fair":
            meets = acc >= self.args.target_acc and gap <= self.args.target_gap + 1e-9
            key = (meets,) + key
        return key

    def compute_loss(self, logits, y, a):
        if self.args.method == "erm":
            return F.cross_entropy(logits, y, weight=self.class_weights)
        if self.args.method == "resample":
            # batches are already balanced over (group x label) cells
            return F.cross_entropy(logits, y)

        # fair: group loss reweighting
        loss = 0.0
        for group, weight in enumerate(self.group_loss_weights):
            mask = a == group
            loss_group = F.cross_entropy(logits[mask], y[mask], weight=self.class_weights)
            loss = loss + weight * loss_group
        return loss

    def update_batch(self, minibatch):
        x = minibatch["img"].to(self.device)
        y = minibatch["label"].to(self.device).long().squeeze(-1)
        a = minibatch["sensitive"].to(self.device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.args.amp):
            logits = self.model(x)
        loss = self.compute_loss(logits.float(), y, a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        logits, targets, sensitive = [], [], []
        for minibatch in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.args.amp):
                out = self.model(minibatch["img"].to(self.device))
            logits.append(out.float().cpu())
            targets.append(minibatch["label"])
            sensitive.append(minibatch["sensitive"])
        logits = torch.concat(logits)
        return (logits.numpy(),
                F.softmax(logits, dim=-1)[:, 1].numpy(),
                torch.concat(targets).squeeze().numpy().astype(int),
                torch.concat(sensitive).squeeze().numpy().astype(int))

    def evaluate(self, dataloader, save_path=None):
        logits, prob, target, sensitive = self.predict(dataloader)
        overall, subgroup = evaluate_binary(prob, target, sensitive)
        organized = organize_results(overall, subgroup)

        summary = {
            "method": self.args.method,
            "run_name": self.args.run_name,
            "age_threshold": self.args.age_threshold,
            "seed": self.args.random_seed,
            "epoch": int(self.epoch),
            "n_test": int(len(target)),
            "group_sizes": [int((sensitive == g).sum()) for g in np.unique(sensitive)],
            # threshold picked by the best F1 on the evaluated set (metric reported in the README)
            "acc@best_f1": float(overall["acc@best_f1"]),
            "group_acc@best_f1": [float(v) for v in subgroup["acc@best_f1"]],
            "acc_gap@best_f1": float(organized["acc-gap"]),
            # fixed 0.5 threshold
            "acc@0.5": float(overall["acc"]),
            "group_acc@0.5": [float(v) for v in subgroup["acc"]],
            "acc_gap@0.5": float(max(subgroup["acc"]) - min(subgroup["acc"])),
            "auc": float(overall["auc"]),
            "group_auc": [float(v) for v in subgroup["auc"]],
        }
        self.logger.info("eval epoch {}: {}".format(self.epoch, json.dumps(summary)))

        if save_path is None:
            # periodic evaluation during training: metric history + best checkpoint (picked on the test set)
            with open(os.path.join(self.args.save_folder, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(summary) + "\n")
            self.eval_probs.append(prob)
            np.savez(os.path.join(self.args.save_folder, "eval_probs.npz"),
                     prob=np.stack(self.eval_probs), label=target, sensitive=sensitive)

            key = self.selection_key(summary)
            if self.best_key is None or key > self.best_key:
                self.best_key = key
                self.save_checkpoint("ckpt_best.pth")
                self.save_results(os.path.join(self.args.save_folder, "best"),
                                  summary, overall, subgroup, logits, target, sensitive)
        else:
            self.save_results(save_path, summary, overall, subgroup, logits, target, sensitive)

        return summary

    def save_results(self, save_path, summary, overall, subgroup, logits, target, sensitive):
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "metrics.pkl"), "wb") as f:
            pickle.dump({"epoch": self.epoch, "overall": overall, "subgroup": subgroup}, f)
        with open(os.path.join(save_path, "predictions.pkl"), "wb") as f:
            pickle.dump({"epoch": self.epoch, "logits": logits, "label": target, "sensitive": sensitive}, f)
        with open(os.path.join(save_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
