"""
losses.py
---------
Focal loss for binary classification with class imbalance handling.

Focal loss (Lin et al. 2017) downweights well-classified easy examples
and focuses training on hard misclassified ones.

Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Why focal over CrossEntropy for this project:
  In early training, clean frames are "easy" — the network quickly learns
  to classify them correctly. Standard CE then wastes gradient on them.
  Focal loss focuses the network on the hard attacked examples where
  stripes are subtle (freq_high_narrow, small coverage).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (works for binary classification with num_classes=2).

    Args:
        alpha : scalar or tensor of shape (num_classes,)
                Per-class weight. Use 0.25 for balanced, higher for minority class.
        gamma : focusing parameter. 0 = standard CE, 2 = standard focal.
        reduction: "mean" | "sum" | "none"
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw unnormalised logits
            targets : (B,) integer class labels
        Returns:
            loss: scalar (if reduction != "none")
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t     = torch.exp(-ce_loss)
        focal   = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def get_loss_fn(cfg: dict, class_weights: torch.Tensor = None) -> nn.Module:
    """
    Build loss function from config.

    Args:
        cfg           : full config dict
        class_weights : optional tensor of shape (2,) for weighted CE
    """
    loss_type = cfg["training"].get("loss", "focal")

    if loss_type == "focal":
        alpha = cfg["training"].get("focal_alpha", 0.25)
        gamma = cfg["training"].get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == "crossentropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'focal' or 'crossentropy'.")
