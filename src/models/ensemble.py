"""
ensemble.py
-----------
Ensemble of specialist CNNs - one per attack variation profile.

Why ensemble beats single CNN:
  Each specialist CNN trains ONLY on clean + one variation type.
  At inference, all specialists vote via probability averaging.
  When an unseen attack arrives (e.g. an in-between frequency/intensity),
  at least a few specialists partially recognise it - the averaged
  probability catches it even when no single model is confident.

WHAT CHANGED FROM THE PREVIOUS VERSION:
Only the __main__ test block. build_ensemble()/save_ensemble()/
load_ensemble() already took `variation_names` as a plain list of
strings, which matches dataloader.get_variation_names() from the
updated dataloader.py - no change needed there. The test block used
to read `cfg["dataset"]["variations"].keys()`, which doesn't exist in
the v7 config (variations are a top-level LIST, not a dict under
`dataset`). Updated to read the same shape get_variation_names() reads.
Note this ensemble is now 5 specialists (freq_low_wide, freq_mid_narrow,
freq_high_fine, freq_ultra_aliasing, freq_random_full), not 6 - "clean"
is excluded since every specialist already trains on clean + its own
variation (see dataloader's variation_filter logic).

Usage:
    from src.models.ensemble import EnsembleCNN, load_ensemble
    ensemble = load_ensemble(variation_names, checkpoint_dir, cfg)
    probs = ensemble.predict_proba(batch)   # (B, 2)
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.models.cnn import LaserAttackCNN, build_model


class EnsembleCNN(nn.Module):
    """
    Ensemble of specialist CNNs.

    Args:
        models    : dict mapping variation_name -> LaserAttackCNN
        strategy  : "average" | "weighted" | "majority_vote"
        weights   : optional dict mapping variation_name -> float weight
                    (used when strategy="weighted")
    """

    def __init__(
        self,
        models: Dict[str, LaserAttackCNN],
        strategy: str = "average",
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.specialist_names = list(models.keys())
        self.specialists = nn.ModuleList(list(models.values()))
        self.strategy = strategy
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits-equivalent averaged probabilities (B, 2).
        """
        return self.predict_proba(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate probability predictions across all specialists.

        Returns:
            probs: (B, 2) averaged probability tensor
        """
        all_probs = []
        for model in self.specialists:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)  # (B, 2)
            all_probs.append(probs)

        if self.strategy == "average":
            stacked = torch.stack(all_probs, dim=0)   # (N_models, B, 2)
            return stacked.mean(dim=0)                 # (B, 2)

        elif self.strategy == "weighted":
            assert self.weights is not None, "Provide weights dict for weighted strategy"
            weight_vals = [self.weights.get(name, 1.0) for name in self.specialist_names]
            weight_tensor = torch.tensor(weight_vals, dtype=torch.float32).to(x.device)
            weight_tensor = weight_tensor / weight_tensor.sum()
            stacked = torch.stack(all_probs, dim=0)   # (N_models, B, 2)
            weighted = (stacked * weight_tensor[:, None, None])
            return weighted.sum(dim=0)                 # (B, 2)

        elif self.strategy == "majority_vote":
            all_preds = [p.argmax(dim=1) for p in all_probs]  # list of (B,)
            votes = torch.stack(all_preds, dim=0).float().mean(dim=0)
            # Return as pseudo-probabilities: [1-vote, vote]
            return torch.stack([1 - votes, votes], dim=1)

        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class (B,): 0=clean, 1=attacked."""
        return self.predict_proba(x).argmax(dim=1)

    def predict_with_specialist_votes(self, x: torch.Tensor):
        """
        Returns full breakdown for debugging/evaluation.

        Returns:
            final_pred : (B,) final ensemble prediction
            spec_preds : dict mapping variation_name -> (B,) specialist prediction
            ensemble_proba: (B, 2) ensemble probabilities
        """
        spec_preds = {}
        for name, model in zip(self.specialist_names, self.specialists):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1).argmax(dim=1)
            spec_preds[name] = pred

        ensemble_proba = self.predict_proba(x)
        final_pred = ensemble_proba.argmax(dim=1)
        return final_pred, spec_preds, ensemble_proba


def build_ensemble(cfg: dict, variation_names: List[str]) -> EnsembleCNN:
    """
    Build an untrained ensemble of specialist CNNs.
    Each specialist has the same architecture (from config["model"]).
    """
    models = {}
    for name in variation_names:
        models[name] = build_model(cfg)  # fresh model per specialist

    strategy = cfg.get("ensemble", {}).get("strategy", "average")
    print(f"  Ensemble: {len(variation_names)} specialists | strategy={strategy}")
    return EnsembleCNN(models=models, strategy=strategy)


def save_ensemble(ensemble: EnsembleCNN, checkpoint_dir: str, variation_names: List[str]):
    """Save each specialist's state dict separately."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    for name, model in zip(variation_names, ensemble.specialists):
        path = os.path.join(checkpoint_dir, f"specialist_{name}.pth")
        torch.save(model.state_dict(), path)
    print(f"  Saved {len(variation_names)} specialist checkpoints to {checkpoint_dir}/")


def load_ensemble(
    variation_names: List[str],
    checkpoint_dir: str,
    cfg: dict,
    device: torch.device = None,
) -> EnsembleCNN:
    """Load a trained ensemble from saved specialist checkpoints."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for name in variation_names:
        path = os.path.join(checkpoint_dir, f"specialist_{name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Run training first: python run.py --mode train --model ensemble"
            )
        model = build_model(cfg)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models[name] = model

    strategy = cfg.get("ensemble", {}).get("strategy", "average")
    return EnsembleCNN(models=models, strategy=strategy)


# -- Quick test ---------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Top-level `variations:` list (v7 schema), excluding the "clean" entry -
    # each specialist already trains on clean + its own variation.
    variation_names = [v["name"] for v in cfg["variations"] if not v.get("clean", False)]
    ensemble = build_ensemble(cfg, variation_names)

    dummy = torch.randn(4, 3, 224, 224)
    probs = ensemble.predict_proba(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {probs.shape}")   # (4, 2)
    print(f"Predictions: {ensemble.predict(dummy)}")
    print("Ensemble model OK")