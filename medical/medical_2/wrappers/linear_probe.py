import torch

from wrappers.base import BaseWrapper


class LinearProbeWrapper(BaseWrapper):
    """Image encoder + linear classification head. The encoder is fine-tuned together with the head."""

    def __init__(self, model) -> None:
        super().__init__(model)

        self.encoder = model
        self.head = torch.nn.Linear(self.encoder.feat_dim, 2)

        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.head(self.encoder(x))
