import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from fairness.dataloader import *
from fairness.model import *
from fairness.metrics_fairness import *
from tqdm import tqdm
import numpy as np
import random

def set_global_seed(seed):
    np.random.seed(seed % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_global_seed(210)
    ckpt_path = "fairness_model_epoch_8.pth"
    test_dataset = loadDataset(dataset='uci_adult', train_or_test="test")

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LearnerNN(
        [(3, 32)], # categorical_embedding_sizes
        6, # len(dataset.mean_std.keys())
        [64, 32], # learner_hidden_units
        activation_fn=nn.ReLU,
        device=device
    )

    model.load_state_dict(torch.load(ckpt_path))

    model.eval().to(device)

    with torch.no_grad():
        test_accuracy = compute_accuracy(model, test_loader, device)
        deo_test = compute_deo(model, test_loader, device)
        print(f"Test accuracy: {test_accuracy}, DEO: {deo_test}")


if __name__ == "__main__":
    main()
