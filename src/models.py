"""
VFL model components for SpectralVFL.
Image client, tabular client, and late-fusion server.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from .config import config


# Backbone output dimensions (after avgpool / global pool, before classifier head)
_BACKBONE_DIMS: Dict[str, int] = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "mobilenet_v3_small": 576,
    "vit_b_16": 768,
    "mamba_vision": 768,
}


def _get_backbone(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Load pretrained backbone and return (model, feature_dim)."""
    model_name = model_name.lower()
    if model_name not in _BACKBONE_DIMS:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Choose from {list(_BACKBONE_DIMS.keys())}."
        )
    dim = _BACKBONE_DIMS[model_name]

    if model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone.classifier = nn.Identity()
    elif model_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone.classifier = nn.Identity()
    elif model_name == "vit_b_16":
        try:
            w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vit_b_16(weights=w)
        except (AttributeError, RuntimeError, TypeError):
            backbone = models.vit_b_16(pretrained=pretrained)
        backbone.heads = nn.Identity()
    elif model_name == "mamba_vision":
        try:
            import timm

            backbone = timm.create_model(
                "mambavision_t_1k",
                pretrained=pretrained,
                num_classes=0,
            )
            dim = int(getattr(backbone, "num_features", dim))
        except Exception:
            w = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.swin_t(weights=w)
            backbone.head = nn.Identity()
            dim = 768
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return backbone, dim


# -----------------------------------------------------------------------------
# Image Client
# -----------------------------------------------------------------------------


class ImageClient(nn.Module):
    """
    Frozen backbone + trainable projection head.
    Outputs fixed-dimension image embeddings for VFL.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        image_emb_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_emb_dim = image_emb_dim

        backbone, backbone_dim = _get_backbone(model_name, pretrained=pretrained)

        # VFL rule: freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, image_emb_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


# -----------------------------------------------------------------------------
# Tabular Client
# -----------------------------------------------------------------------------


class TabularClient(nn.Module):
    """
    Trainable MLP for tabular metadata.
    Outputs fixed-dimension tabular embeddings for VFL.
    """

    def __init__(
        self,
        input_dim: int,
        tab_emb_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.tab_emb_dim = tab_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tab_emb_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# -----------------------------------------------------------------------------
# VFL Server
# -----------------------------------------------------------------------------


class VFLServer(nn.Module):
    """
    Late-fusion MLP over concatenated client embeddings.
    Outputs classification logits.
    """

    def __init__(
        self,
        image_emb_dim: int = 128,
        tab_emb_dim: int = 64,
        num_classes: int = 7,
    ):
        super().__init__()
        input_dim = image_emb_dim + tab_emb_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_emb: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([image_emb, tab_emb], dim=1)
        return self.fusion(combined)


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def get_vfl_system(
    model_name: str = "resnet50",
    tabular_dim: int = 19,
    image_emb_dim: int = 128,
    tab_emb_dim: int = 64,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
) -> Tuple[ImageClient, TabularClient, VFLServer]:
    """
    Initialize and return the full VFL system.

    Args:
        model_name: Backbone for ImageClient (
            'resnet50', 'efficientnet_b0', 'mobilenet_v3_small', 'vit_b_16', 'mamba_vision'
        ).
        tabular_dim: Input dimension for TabularClient (from TabularPreprocessor.tabular_dim).
        image_emb_dim: Output dimension of ImageClient.
        tab_emb_dim: Output dimension of TabularClient.
        num_classes: Number of output classes (default: 7 for HAM10000).
        pretrained: Whether to load pretrained backbone weights.

    Returns:
        (image_client, tabular_client, vfl_server)
    """
    num_classes = num_classes or len(config.tabular.dx_classes)

    image_client = ImageClient(
        model_name=model_name,
        image_emb_dim=image_emb_dim,
        pretrained=pretrained,
    )
    tabular_client = TabularClient(
        input_dim=tabular_dim,
        tab_emb_dim=tab_emb_dim,
    )
    vfl_server = VFLServer(
        image_emb_dim=image_emb_dim,
        tab_emb_dim=tab_emb_dim,
        num_classes=num_classes,
    )

    return image_client, tabular_client, vfl_server
