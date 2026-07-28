"""
losses.py
---------
Loss function factory for train.py. This file didn't exist yet - train.py
imports `get_loss_fn` from here, so training would ImportError without it.

Supports two loss types, selected via cfg["training"]["loss_type"]:
  - "weighted_ce" (default): standard cross-entropy, weighted by inverse
    class frequency (from LaserAttackDataset.get_class_weights()) to handle
    any train-split imbalance between clean/attacked samples.
  - "focal": focal loss, which down-weights easy (already-confident)
    examples and focuses gradient on hard ones - useful if the model starts
    getting very confident on easy variations (e.g. freq_ultra_aliasing,
    which tends to have very high peak_saturation) while still missing
    harder/subtler ones (e.g. freq_low_wide's broad, soft bands).
    cfg["training"]["focal_gamma"] controls focusing strength (default 2.0).

Usage:
    loss_fn = get_loss_fn(cfg, class_weights_tensor)
    loss = loss_fn(logits, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017), reduces to weighted CE when gamma=0."""

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)  # model's predicted probability for the true class
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce).mean()


def get_loss_fn(cfg: dict, class_weights: torch.Tensor = None) -> nn.Module:
    """Build the loss function from config.

    Args:
        cfg           : full config dict
        class_weights : (2,) tensor [weight_clean, weight_attacked], typically
                         from LaserAttackDataset.get_class_weights(). Pass
                         None to disable weighting (equal weight per class).
    """
    train_cfg = cfg.get("training", {})
    loss_type = train_cfg.get("loss_type", "weighted_ce")

    if loss_type == "focal":
        gamma = train_cfg.get("focal_gamma", 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'weighted_ce' or 'focal'.")


# -- Quick test ----------------------------------------------------------------
if __name__ == "__main__":
    cfg = {"training": {"loss_type": "weighted_ce"}}
    weights = torch.tensor([1.2, 0.9])
    loss_fn = get_loss_fn(cfg, weights)
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    loss = loss_fn(logits, labels)
    print("weighted_ce loss:", loss.item())

    cfg["training"]["loss_type"] = "focal"
    loss_fn = get_loss_fn(cfg, weights)
    loss = loss_fn(logits, labels)
    print("focal loss:", loss.item())
    print("OK")