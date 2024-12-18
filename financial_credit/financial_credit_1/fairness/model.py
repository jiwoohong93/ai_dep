import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnerNN(nn.Module):
    def __init__(
        self, embedding_size, n_num_cols, n_hidden, 
        activation_fn=nn.ReLU, device='cpu'
    ):
        """
        Implements the learner DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                           embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
          activation_fn: the activation function to use.
        """
        super().__init__()
        self.device = device

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(n_classes, n_features)
                for n_classes, n_features in embedding_size
            ]
        )

        all_layers = []
        n_cat_cols = sum((n_features for _, n_features in embedding_size))
        input_size = n_cat_cols + n_num_cols

        for dim in n_hidden:
            all_layers.append(nn.Linear(input_size, dim))
            all_layers.append(activation_fn())
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], 1))

        self.layers = nn.Sequential(*all_layers)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_cat, x_num):
        """
        The forward step for the learner.
        """

        embedding_cols = []
        for i, emb in enumerate(self.embeddings):
            embedding_cols.append(emb(x_cat[:, i]))

        x = torch.cat(embedding_cols, dim=1)
        x = torch.cat([x, x_num], dim=1)

        # Get the logits output (for calculating loss)
        logits = self.layers(x)

        sigmoid_output = self.sigmoid(logits)

        sigmoid_output.to(self.device)
        class_predictions = torch.where(
            sigmoid_output > 0.5,
            torch.tensor(1, dtype=torch.float32).to(self.device),
            torch.tensor(0, dtype=torch.float32).to(self.device),
        )

        return logits, sigmoid_output, class_predictions
