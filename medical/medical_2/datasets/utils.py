import random
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler

import datasets


class FairBatchSampler(Sampler):
    """Fair batch: every batch holds a fixed share of each sensitive group.

    target_ratios maps group label -> share of the batch, e.g. {0: 0.5, 1: 0.5}.
    Each group is cycled through in dataset order, so the smaller group is oversampled.
    """

    def __init__(self, dataset, batch_size, target_ratios):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_ratios = target_ratios

        sensitive = np.array([dataset.get_sensitive(idx) for idx in range(len(dataset))])
        self.group_iters = {
            int(group): cycle(np.nonzero(sensitive == group)[0].tolist()) for group in np.unique(sensitive)
        }

    def __iter__(self):
        for _ in range(len(self)):
            batch = []
            for group, ratio in self.target_ratios.items():
                batch.extend(next(self.group_iters[group]) for _ in range(int(self.batch_size * ratio)))
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


class CellBalancedSampler(Sampler):
    """Every batch holds the same number of images from each (sensitive group x label) cell."""

    def __init__(self, dataset, batch_size, generator):
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generator
        cells = np.array([2 * dataset.get_sensitive(i) + int(dataset.Y[i]) for i in range(len(dataset))])
        self.cell_indices = [np.nonzero(cells == c)[0] for c in np.unique(cells)]

    def __iter__(self):
        per_cell = self.batch_size // len(self.cell_indices)
        for _ in range(len(self)):
            batch = []
            for indices in self.cell_indices:
                pick = torch.randint(len(indices), (per_cell,), generator=self.generator)
                batch.extend(indices[pick.numpy()].tolist())
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


def get_transform(args, split):
    size = args.img_size
    normalize = transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    if split == "train" and args.augment:
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataset(args, split):
    data_setting = args.data_setting

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    def seed_worker(worker_id):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    meta = pd.read_csv(data_setting[f"{split}_meta_path"])
    meta["Age_binary"] = (meta["Age"] >= args.age_threshold).astype(int)
    data = getattr(datasets, args.dataset)(
        meta, args.sensitive_name, get_transform(args, split), path_to_images=data_setting["image_path"])
    data.view, data.img_size = args.view, args.img_size

    loader_kwargs = dict(num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g, pin_memory=True)

    if split == "train" and args.method == "fair":
        # FairFundusNet: fixed share of each sensitive group in every batch (default half / half)
        ratio = args.fair_batch_ratio
        sampler = FairBatchSampler(data, batch_size=args.batch_size, target_ratios={0: 1 - ratio, 1: ratio})
        data_loader = DataLoader(data, batch_sampler=sampler, **loader_kwargs)
    elif split == "train" and args.method == "resample":
        sampler = CellBalancedSampler(data, batch_size=args.batch_size, generator=g)
        data_loader = DataLoader(data, batch_sampler=sampler, **loader_kwargs)
    elif split == "train":
        # baseline: ordinary shuffled batches
        data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    else:
        data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    return data, data_loader
