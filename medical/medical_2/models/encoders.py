import timm
import torch
import torch.nn as nn

IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class TimmViT(nn.Module):
    """timm ViT encoder; feature = [CLS token, mean of patch tokens]."""

    def __init__(self, arch, **kwargs) -> None:
        super().__init__()
        self.model = timm.create_model(arch, pretrained=True, num_classes=0, **kwargs)
        self.feat_dim = 2 * self.model.num_features
        self.mean, self.std = IMAGENET_NORM

    def forward(self, images):
        tokens = self.model.forward_features(images)
        return torch.cat([tokens[:, 0], tokens[:, self.model.num_prefix_tokens:].mean(1)], dim=1)


def build_encoder(name):
    if name == "biomedclip":
        from models.biomed_clip import BiomedCLIP
        return BiomedCLIP()
    if name == "dinov2_b":
        return TimmViT("vit_base_patch14_dinov2.lvd142m", dynamic_img_size=True)
    if name == "dinov2_l":
        return TimmViT("vit_large_patch14_dinov2.lvd142m", dynamic_img_size=True)
    raise ValueError(name)
