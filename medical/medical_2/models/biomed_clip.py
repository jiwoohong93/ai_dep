import torch.nn as nn
from open_clip import create_model_from_pretrained


class BiomedCLIP(nn.Module):
    """Image encoder (ViT-B/16) of BiomedCLIP."""

    def __init__(self) -> None:
        super().__init__()
        self.model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        self.feat_dim = 512
        self.mean, self.std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]

    def forward(self, images):
        return self.model.visual(images)
