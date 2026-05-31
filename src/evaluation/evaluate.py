"""
evaluate.py
-----------
Full evaluation pipeline.

Computes:
  - F1, AUC-ROC, Accuracy, Precision, Recall
  - Confusion matrix
  - Per-variation breakdown (which attack type is hardest to detect?)
  - Single CNN vs Ensemble comparison table

Usage:
    python src/evaluation/evaluate.py --model single --checkpoint checkpoints/single_cnn_best.pth
    python src/evaluation/evaluate.py --model ensemble --checkpoint_dir checkpoints/
    OR via run.py:
    python run.py --mode evaluate --model ensemble
    python run.py --mode compare
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import yaml
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.dataloader import get_dataloaders, get_variation_names, LaserAttackDataset, get_transforms
from src.models.cnn import build_model
from src.models.ensemble import load_ensemble
from src.utils.logger import get_logger

logger = get_logger("evaluator")


def evaluate_model(model, loader, device: torch.device, is_ensemble: bool = False) -> dict:
    """
    Run model on a DataLoader and return all metrics.

    Returns dict with keys: f1, auc, accuracy, precision, recall,
                             confusion_matrix, all_preds, all_labels, all_probs
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if is_ensemble:
                probs = model.predict_proba(imgs).cpu()
            else:
                probs = torch.softmax(model(imgs), dim=1).cpu()

            preds = probs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:, 1].tolist())  # probability of class=1 (attacked)

    f1        = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc       = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "f1":               f1,
        "auc":              auc,
        "accuracy":         acc,
        "precision":        precision,
        "recall":           recall,
        "confusion_matrix": cm,
        "all_preds":        all_preds,
        "all_labels":       all_labels,
        "all_probs":        all_probs,
    }


def evaluate_per_variation(model, cfg: dict, split: str, device: torch.device, is_ensemble: bool) -> dict:
    """
    Evaluate model separately on each attack variation.
    Shows which variation is hardest to detect.
    """
    variation_names = get_variation_names(cfg)
    image_size      = cfg["dataset"]["image_size"]
    dataset_root    = cfg["paths"]["final_dataset"]
    csv_path        = os.path.join(dataset_root, "labels.csv")
    batch_size      = cfg["training"]["batch_size"]
    tfm             = get_transforms(split, image_size)

    results = {}
    for var_name in variation_names:
        dataset = LaserAttackDataset(
            csv_path         = csv_path,
            dataset_root     = dataset_root,
            split            = split,
            transform        = tfm,
            variation_filter = var_name,
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        metrics = evaluate_model(model, loader, device, is_ensemble)
        results[var_name] = metrics
        logger.info(f"  {var_name:<25} F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}")

    return results


def print_full_report(metrics: dict, model_name: str, split: str):
    """Pretty-print evaluation results."""
    logger.info("\n" + "="*65)
    logger.info(f"EVALUATION RESULTS — {model_name} on {split} split")
    logger.info("="*65)
    logger.info(f"  F1 Score   : {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC    : {metrics['auc']:.4f}")
    logger.info(f"  Accuracy   : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision  : {metrics['precision']:.4f}")
    logger.info(f"  Recall     : {metrics['recall']:.4f}")
    logger.info(f"  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    logger.info(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    logger.info(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    logger.info("="*65)


def compare_single_vs_ensemble(cfg: dict):
    """
    Run both single CNN and ensemble on the test set and print comparison.
    Requires both to be trained first.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(cfg, variation_filter=None)
    variation_names = get_variation_names(cfg)
    ckpt_dir = cfg["paths"]["checkpoints"]

    results = {}

    # ── Single CNN ─────────────────────────────────────────────────────────
    single_path = os.path.join(ckpt_dir, "single_cnn_best.pth")
    if os.path.exists(single_path):
        logger.info("\nEvaluating Single CNN...")
        single_model = build_model(cfg)
        single_model.load_state_dict(torch.load(single_path, map_location=device))
        single_model.to(device).eval()
        m = evaluate_model(single_model, loaders["test"], device, is_ensemble=False)
        results["Single CNN"] = m
        print_full_report(m, "Single CNN", "test")
    else:
        logger.warning(f"Single CNN checkpoint not found at {single_path}")

    # ── Ensemble ───────────────────────────────────────────────────────────
    try:
        logger.info("\nEvaluating Ensemble CNN...")
        ensemble = load_ensemble(variation_names, ckpt_dir, cfg, device)
        m = evaluate_model(ensemble, loaders["test"], device, is_ensemble=True)
        results["Ensemble"] = m
        print_full_report(m, "Ensemble CNN", "test")
    except FileNotFoundError as e:
        logger.warning(str(e))

    # ── Comparison Table ───────────────────────────────────────────────────
    if len(results) == 2:
        logger.info("\n" + "="*65)
        logger.info("COMPARISON TABLE")
        logger.info("="*65)
        logger.info(f"{'Metric':<20} {'Single CNN':>15} {'Ensemble':>15} {'Winner':>10}")
        logger.info("-"*65)
        for metric in ["f1", "auc", "accuracy", "precision", "recall"]:
            s = results["Single CNN"][metric]
            e = results["Ensemble"][metric]
            winner = "Ensemble ✓" if e >= s else "Single"
            logger.info(f"{metric:<20} {s:>15.4f} {e:>15.4f} {winner:>10}")
        logger.info("="*65)

    # Save results
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "comparison.json")
    serialisable = {
        k: {mk: mv for mk, mv in v.items() if mk not in ("all_preds", "all_labels", "all_probs")}
        for k, v in results.items()
    }
    with open(save_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"\nResults saved to {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="ensemble",  choices=["single", "ensemble", "compare"])
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--checkpoint",    default=None, help="Path for single model .pth")
    parser.add_argument("--checkpoint_dir",default=None, help="Directory for ensemble checkpoints")
    parser.add_argument("--split",         default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model == "compare":
        compare_single_vs_ensemble(cfg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaders = get_dataloaders(cfg, variation_filter=None)
        variation_names = get_variation_names(cfg)
        ckpt_dir = args.checkpoint_dir or cfg["paths"]["checkpoints"]

        if args.model == "single":
            ckpt = args.checkpoint or os.path.join(ckpt_dir, "single_cnn_best.pth")
            model = build_model(cfg)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device).eval()
            m = evaluate_model(model, loaders[args.split], device)
            print_full_report(m, "Single CNN", args.split)

            logger.info("\nPer-variation breakdown:")
            evaluate_per_variation(model, cfg, args.split, device, is_ensemble=False)

        else:
            ensemble = load_ensemble(variation_names, ckpt_dir, cfg, device)
            m = evaluate_model(ensemble, loaders[args.split], device, is_ensemble=True)
            print_full_report(m, "Ensemble CNN", args.split)

            logger.info("\nPer-variation breakdown:")
            evaluate_per_variation(ensemble, cfg, args.split, device, is_ensemble=True)
