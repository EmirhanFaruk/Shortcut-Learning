"""
models.py
---------
Two backbones so Phase 1 can be a small architecture comparison.
Both use ImageNet-pretrained weights and a fresh 2-way head.
"""
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int = 2, pretrained: bool = True):
    name = name.lower()
    if name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1]
    elif name in {"efficientnet", "efficientnet_b0", "effnet"}:
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        target_layer = m.features[-1]
    else:
        raise ValueError(f"unknown model: {name}")
    return m, target_layer
