import os

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch

from utils.lr_sched import adjust_learning_rate


class BaseTrainer(object):
    def __init__(self, args, model, logger) -> None:
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.logger = logger

        self.epoch = 0
        self.total_epochs = args.total_epochs

    def init_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.args.weight_decay)
        if self.args.head_lr_scale != 1.0:
            # separate, faster-learning groups for the classification head
            head = [p for p in self.model.head.parameters()]
            head_ids = {id(p) for p in head}
            for group in param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in head_ids]
            param_groups.append({"params": head, "weight_decay": 0.0, "lr_scale": self.args.head_lr_scale})
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(param_groups, lr=self.args.lr)
        elif self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr)
        else:
            raise NotImplementedError

    def train(self, train_dataloader):
        while self.epoch < self.total_epochs:
            self.model.train()
            adjust_learning_rate(self.optimizer, self.epoch + 1, self.args)

            loss = self.train_epoch(train_dataloader)
            self.logger.info("epoch {}: lr {}, loss {}".format(
                self.epoch, self.optimizer.param_groups[0]["lr"], loss))

            if self.epoch % self.args.eval_freq == 0:
                self.evaluate(self.test_loader)

            self.epoch += 1

        self.save_checkpoint("ckpt_last.pth")

    def train_epoch(self, train_dataloader):
        loss_epoch = [self.update_batch(minibatch).item() for minibatch in train_dataloader]
        return np.mean(loss_epoch)

    def save_checkpoint(self, name):
        path = os.path.join(self.args.save_folder, name)
        torch.save({"model": self.model.state_dict(), "epoch": self.epoch, "method": self.args.method},
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.epoch = ckpt.get("epoch", 0)

    def update_batch(self, minibatch):
        raise NotImplementedError

    def evaluate(self, dataloader, save_path=None):
        raise NotImplementedError
