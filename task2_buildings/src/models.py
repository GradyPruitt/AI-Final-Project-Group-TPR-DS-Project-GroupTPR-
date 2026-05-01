"""
Deep learning models for DoriaNET building damage classification.

The final exam requires THREE deep computer vision architectures. We chose
three families that cover meaningful design diversity:

  1. ResNet-50           — convolutional, residual blocks, ImageNet-pretrained.
                          The classical CNN baseline. Used widely as the
                          "safe choice" backbone in computer vision.

  2. EfficientNet-B0     — convolutional, scaled depth/width/resolution
                          via compound scaling. State-of-the-art-CNN with
                          better accuracy-per-parameter than ResNet.

  3. Vision Transformer  — non-convolutional, treats image as a sequence
     (ViT-B/16)            of 16×16 patches and applies self-attention.
                          Different inductive bias than CNNs entirely.

Why these three?
----------------
A meaningful comparison needs architectures with different inductive biases
or capacity profiles. ResNet-50 vs EfficientNet-B0 lets us compare two CNN
families. EfficientNet vs ViT-B/16 lets us compare CNN vs Transformer.
ResNet vs ViT is the canonical "old vs new" CNN-vs-Transformer comparison
that's been studied extensively in the literature.

All three are loaded from `torchvision.models` with ImageNet pretrained
weights. We replace each model's final classification head with a small
2-layer MLP that outputs `n_classes` (= 6 damage levels). Transfer learning
is essential here — DoriaNET is small (~1000 training samples), so training
from scratch would severely underperform.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Shared classification head
# ---------------------------------------------------------------------------

def _make_head(in_features: int, n_classes: int, dropout: float = 0.3) -> nn.Module:
    """A small classification head. Same shape across all three models so the
    only thing different between them is the backbone."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, n_classes),
    )


# ---------------------------------------------------------------------------
# 1. ResNet-50  (the classical CNN baseline)
# ---------------------------------------------------------------------------

class ResNet50Classifier(nn.Module):
    """ResNet-50 backbone with replaced classification head.

    ResNet-50 has 25M parameters in the backbone. We initialize from ImageNet
    pretrained weights (the standard transfer-learning approach for small
    datasets like DoriaNET).

    By default we freeze the first three "layer1/layer2/layer3" blocks
    and only fine-tune layer4 + the head. This is a common middle-ground
    between full fine-tuning (overfit-prone) and frozen-features (under-
    learns task-specific patterns).
    """

    def __init__(self, n_classes: int = 6, freeze_early: bool = True, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features

        # Replace the head
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = _make_head(in_features, n_classes, dropout=dropout)

        if freeze_early:
            for name, p in self.backbone.named_parameters():
                if name.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.")):
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (B, 2048)
        return self.head(feats)         # (B, n_classes)


# ---------------------------------------------------------------------------
# 2. EfficientNet-B0  (modern CNN with compound scaling)
# ---------------------------------------------------------------------------

class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 backbone with replaced classification head.

    EfficientNet-B0 has ~5M backbone parameters — five times smaller than
    ResNet-50, but Tan & Le (2019) showed it matches or exceeds ResNet
    accuracy on ImageNet. Useful comparison for "does smaller and better-
    designed beat bigger" on this small dataset.
    """

    def __init__(self, n_classes: int = 6, freeze_early: bool = True, dropout: float = 0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # The classifier in torchvision's EfficientNet is a Sequential
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = _make_head(in_features, n_classes, dropout=dropout)

        if freeze_early:
            # Freeze the first 4 of 9 "features" blocks
            for i, block in enumerate(self.backbone.features):
                if i < 4:
                    for p in block.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (B, 1280)
        return self.head(feats)


# ---------------------------------------------------------------------------
# 3. Vision Transformer (ViT-B/16) — non-convolutional alternative
# ---------------------------------------------------------------------------

class ViTClassifier(nn.Module):
    """ViT-B/16 backbone with replaced classification head.

    ViT-B/16 has 86M parameters and treats the image as a sequence of
    16x16 patches that pass through a Transformer encoder. Very different
    inductive bias than CNNs — no convolutions, no built-in translation
    invariance, learns spatial relationships purely through self-attention.

    On small datasets like ours, ViT typically requires more careful
    training than CNNs and can be more prone to overfitting. The trade-off
    is that ViT often captures longer-range dependencies better when there's
    enough data.
    """

    def __init__(self, n_classes: int = 6, freeze_early: bool = True, dropout: float = 0.3):
        super().__init__()
        backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # ViT's classification head is `heads.head`, a single Linear layer
        in_features = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.head = _make_head(in_features, n_classes, dropout=dropout)

        if freeze_early:
            # Freeze the first 8 of 12 encoder blocks; fine-tune the last 4.
            # ViT's torchvision API exposes encoder.layers as a ModuleList.
            for i, layer in enumerate(self.backbone.encoder.layers):
                if i < 8:
                    for p in layer.parameters():
                        p.requires_grad = False
            # Also freeze patch embedding so we don't degrade the input stem
            for p in self.backbone.conv_proj.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (B, 768)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "resnet50":     ResNet50Classifier,
    "efficientnet": EfficientNetB0Classifier,
    "vit":          ViTClassifier,
}


def build_model(name: str, n_classes: int = 6, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_classes=n_classes, **kwargs)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
