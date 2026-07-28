"""
train.py
--------
Full training pipeline.

Supports:
  --model single    : Train one CNN on all 5 attack variations + clean combined
  --model ensemble  : Train 5 specialist CNNs, one per attack variation

WHAT CHANGED FROM THE PREVIOUS VERSION:
1. Path resolution fix: cfg["paths"]["checkpoints"] and cfg["paths"]["results"]
   are written relative to the project root (same convention as
   paths.clean_base/final_dataset everywhere else in this pipeline), but the
   previous version passed them straight to os.path.join() without resolving
   against project root - meaning checkpoints would land in whatever the cwd
   happened to be when you ran the script, same class of bug fixed earlier
   in dataloader.py and dataset_builder.py. Fixed the same way: resolve
   relative to this file's own location.
2. losses.py and utils/logger.py didn't exist - both created (see those
   files). Nothing in this file changed to accommodate them beyond the
   existing imports already being correct.
3. Comment/log text said "6 variations" - your config has 5 attack
   variations (clean is not a specialist, it's included in every specialist's
   training data via dataloader's variation_filter logic) - updated to match.

Features:
  - Early stopping (patience from config)
  - Best model checkpointing (val F1, not just val loss)
  - LR scheduler (cosine annealing)
  - Mixed precision (AMP) for faster GPU training
  - Detailed per-epoch logging
  - Training curve saved to results/

Usage:
    python src/training/train.py --model single
    python src/training/train.py --model ensemble
    OR via run.py:
    python run.py --mode train --model single
    python run.py --mode train --model ensemble
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml
try:
    # torch >= 2.4 API
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
    def _make_scaler(enabled): return GradScaler(_AMP_DEVICE, enabled=enabled)
    def _autocast(enabled): return autocast(_AMP_DEVICE, enabled=enabled)
except ImportError:
    # older torch fallback
    from torch.cuda.amp import GradScaler, autocast
    def _make_scaler(enabled): return GradScaler(enabled=enabled)
    def _autocast(enabled): return autocast(enabled=enabled)
from sklearn.metrics import f1_score

# Project root, derived from this file's own location (src/training/train.py
# -> project root is two levels up). Used both for the sys.path insert (so
# `from src.dataset...` imports work regardless of cwd) and for resolving
# paths.checkpoints/paths.results the same way dataloader.py resolves
# paths.final_dataset - those are written relative to project root in
# config.yaml, not relative to wherever this script happens to be invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.dataloader import get_dataloaders, get_variation_names
from src.models.cnn import build_model
from src.models.ensemble import build_ensemble, save_ensemble
from src.training.losses import get_loss_fn
from src.utils.logger import get_logger

logger = get_logger("trainer")


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE CNN TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_single(cfg: dict) -> dict:
    """Train one CNN on all variations combined. Returns best val metrics."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODE: Single CNN (all variations)")
    logger.info("="*60)

    device = _get_device()
    loaders = get_dataloaders(cfg, variation_filter=None)

    model    = build_model(cfg).to(device)
    loss_fn  = get_loss_fn(cfg, loaders["train"].dataset.get_class_weights().to(device))
    results  = _run_training(model, loaders, loss_fn, cfg, device, tag="single")

    # Save final model
    ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "single_cnn_best.pth")
    logger.info(f"Saved: {ckpt_dir}/single_cnn_best.pth")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_ensemble(cfg: dict) -> dict:
    """Train one specialist CNN per attack variation. Returns aggregated metrics."""
    device = _get_device()
    variation_names = get_variation_names(cfg)

    logger.info("\n" + "="*60)
    logger.info(f"TRAINING MODE: Ensemble ({len(variation_names)} specialist CNNs)")
    logger.info("="*60)

    all_results = {}

    for i, var_name in enumerate(variation_names):
        logger.info(f"\n--- Specialist {i+1}/{len(variation_names)}: {var_name} ---")
        loaders = get_dataloaders(cfg, variation_filter=var_name)
        model   = build_model(cfg).to(device)
        loss_fn = get_loss_fn(cfg, loaders["train"].dataset.get_class_weights().to(device))
        results = _run_training(model, loaders, loss_fn, cfg, device, tag=f"specialist_{var_name}")

        # Save specialist checkpoint
        ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoints"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"specialist_{var_name}.pth")
        all_results[var_name] = results

    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE TRAINING COMPLETE - Specialist Summary")
    logger.info("="*60)
    logger.info(f"{'Variation':<25} {'Best Val F1':>12} {'Best Val Acc':>12}")
    logger.info("-"*50)
    for var_name, res in all_results.items():
        logger.info(f"{var_name:<25} {res['best_val_f1']:>12.4f} {res['best_val_acc']:>12.4f}")

    avg_f1 = sum(r["best_val_f1"] for r in all_results.values()) / len(all_results)
    logger.info("-"*50)
    logger.info(f"{'Average':<25} {avg_f1:>12.4f}")
    logger.info("="*60)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# CORE TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def _run_training(
    model: nn.Module,
    loaders: dict,
    loss_fn: nn.Module,
    cfg: dict,
    device: torch.device,
    tag: str,
) -> dict:
    """Inner training loop shared by single and ensemble modes."""
    train_cfg = cfg["training"]
    epochs    = train_cfg["epochs"]
    patience  = train_cfg["patience"]
    use_amp   = train_cfg.get("amp", True) and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg["learning_rate"],
        weight_decay = train_cfg["weight_decay"],
    )

    scheduler = _build_scheduler(optimizer, train_cfg, epochs)
    scaler    = _make_scaler(use_amp)

    best_val_f1  = 0.0
    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}  # fallback if epoch 1 never improves
    no_improve   = 0
    history      = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # -- Train --------------------------------------------------------------
        model.train()
        train_loss = 0.0
        for imgs, labels in loaders["train"]:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with _autocast(use_amp):
                logits = model(imgs)
                loss   = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(loaders["train"])

        # -- Validate -------------------------------------------------------------
        val_loss, val_f1, val_acc = _evaluate_epoch(model, loaders["val"], loss_fn, device, use_amp)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        # -- Checkpoint on best val F1 --------------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  New best val F1: {best_val_f1:.4f}")
        else:
            no_improve += 1

        # -- Early stopping ---------------------------------------------------------
        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    logger.info(f"\n  Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")

    # Save training history
    _save_history(history, PROJECT_ROOT / cfg["paths"]["results"], tag)

    return {"best_val_f1": best_val_f1, "best_val_acc": best_val_acc, "best_epoch": best_epoch}


def _evaluate_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Evaluate model for one epoch. Returns (loss, f1, accuracy)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with _autocast(use_amp):
                logits = model(imgs)
                loss   = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, f1, acc


def _build_scheduler(optimizer, train_cfg: dict, epochs: int):
    sched_type = train_cfg.get("scheduler", "cosine")
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    return None


def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("  Running on CPU - consider using GPU for faster training")
    return device


def _save_history(history: dict, results_dir, tag: str):
    import json
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"history_{tag}.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# -- CLI ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="ensemble", choices=["single", "ensemble"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model == "single":
        train_single(cfg)
    else:
        train_ensemble(cfg)