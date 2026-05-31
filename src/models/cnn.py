"""
cnn.py
------
Single CNN model for laser dazzle attack detection.

Architecture:
  - Backbone: ResNet18 (pretrained on ImageNet)
  - Head: Dropout → Linear(512 → 256) → ReLU → Linear(256 → 2)
  - Output: 2-class logits (clean / attacked)

Why ResNet18:
  - Fast to train on CPU (30–60 min for 30 epochs)
  - Strong pretrained features for texture/pattern detection
  - Easy to swap for ResNet34 or EfficientNet-B0 via config

Usage:
    from src.models.cnn import build_model
    model = build_model(cfg)
"""

import torch
import torch.nn as nn
from torchvision import models


class LaserAttackCNN(nn.Module):
    """
    Binary CNN: clean (0) vs attacked (1).

    Args:
        backbone   : "resnet18" | "resnet34" | "efficientnet_b0"
        pretrained : use ImageNet pretrained weights
        dropout    : dropout rate before classifier head
        num_classes: 2 (binary)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.backbone_name = backbone

        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            base = models.resnet18(weights=weights)
            in_features = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=weights)
            in_features = 512
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            in_features = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose resnet18, resnet34, or efficientnet_b0.")

        # Remove original classifier
        if "resnet" in backbone:
            self.features = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        else:
            self.features = base.features
            self.avgpool  = nn.AdaptiveAvgPool2d(1)

        # Custom binary head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """He initialisation for the custom head layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised image tensor
        Returns:
            logits: (B, 2)
        """
        feats = self.features(x)
        if hasattr(self, "avgpool"):
            feats = self.avgpool(feats)
        feats = feats.flatten(1)
        return self.classifier(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (B, 2)."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class (B,): 0=clean, 1=attacked."""
        return self.predict_proba(x).argmax(dim=1)


def build_model(cfg: dict) -> LaserAttackCNN:
    """Build model from config dict."""
    model_cfg = cfg["model"]
    model = LaserAttackCNN(
        backbone    = model_cfg["backbone"],
        pretrained  = model_cfg["pretrained"],
        dropout     = model_cfg["dropout"],
        num_classes = model_cfg["num_classes"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_cfg['backbone']} | Trainable params: {n_params:,}")
    return model


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")    # (4, 2)
    print(f"Probabilities:\n{torch.softmax(out, dim=1)}")
    print("CNN model OK ✓")
