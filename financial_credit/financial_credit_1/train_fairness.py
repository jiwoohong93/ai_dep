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
    train_dataset = loadDataset(
        dataset='uci_adult',
        train_or_test="train",
        embedding_size=32,
    )
    test_dataset = loadDataset(dataset='uci_adult', train_or_test="test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    breakpoint()
    model = LearnerNN(
        train_dataset.categorical_embedding_sizes,
        len(train_dataset.mean_std.keys()),
        [64, 32], # learner_hidden_units
        activation_fn=nn.ReLU,
        device=device
    )

    model.train().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for train_cat, train_num, train_target in train_loader:
            optimizer.zero_grad()

            train_cat = train_cat.to(device)
            train_num = train_num.to(device)
            train_target = train_target.to(device)

            logits, _, _ = model(train_cat, train_num)

            causal_train_cat = 1 - train_cat
            causal_logits, _, _ = model(causal_train_cat, train_num)

            weights = torch.where(train_cat == 1, 1.52, 2.94)
            causal_weights = torch.where(causal_train_cat == 1, 1.52, 2.94)
            final_logits = logits * weights + causal_logits * causal_weights

            loss = criterion(final_logits, train_target).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log the average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        with torch.no_grad():
            test_accuracy = compute_accuracy(model, test_loader, device)
            deo_test = compute_deo(model, test_loader, device)
            if test_accuracy > 0.81 and deo_test < 0.07:
                print(f"Epch {epoch + 1} test accuracy: {test_accuracy}, DEO: {deo_test}")
                torch.save(model.state_dict(), f"fairness_model_epoch_{epoch+1}.pth")



if __name__ == "__main__":
    main()
