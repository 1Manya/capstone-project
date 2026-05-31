"""
dataset_builder.py
------------------
Full dataset generation pipeline.

1. Scans data/clean_base/ for all images
2. Resizes to config image_size
3. Saves clean images as label=0
4. For each of 6 variation profiles, injects stripes → label=1
5. Splits into train / val / test (stratified)
6. Writes data/final_dataset/labels.csv

Run:
    python src/dataset/dataset_builder.py
    OR
    python run.py --mode generate
"""

import os
import sys
import csv
import shutil
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.stripe_generator import inject_stripes
from src.utils.logger import get_logger

logger = get_logger("dataset_builder")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_images(clean_base: str, max_images: int) -> list:
    """Return sorted list of image paths from clean_base directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for root, _, files in os.walk(clean_base):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(root, f))

    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError(
            f"\n\nNo images found in '{clean_base}'.\n"
            "Please put KITTI images (or any driving scene images) in that folder.\n"
            "KITTI download: http://www.cvlibs.net/datasets/kitti/eval_object.php\n"
            "→ 'Left color images of object data set'"
        )

    if len(paths) > max_images:
        random.shuffle(paths)
        paths = paths[:max_images]
        logger.info(f"Capped at {max_images} images (set max_images in config to change)")

    logger.info(f"Found {len(paths)} clean base images in '{clean_base}'")
    return paths


def stratified_split(paths: list, train: float, val: float, seed: int):
    """Split list into train/val/test maintaining approximate ratios."""
    random.seed(seed)
    shuffled = paths.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:]
    )


def process_and_save(
    src_path: str,
    dest_path: str,
    image_size: int,
    variation_cfg: dict = None,
    seed: int = None,
) -> bool:
    """
    Read image, optionally inject stripes, resize, save.
    Returns True on success.
    """
    img = cv2.imread(src_path)
    if img is None:
        logger.warning(f"Could not read {src_path}, skipping.")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if variation_cfg is not None:
        img_rgb = inject_stripes(img_rgb, variation_cfg, seed=seed)

    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cv2.imwrite(dest_path, img_bgr)
    return True


def build_dataset(config_path: str = "configs/config.yaml"):
    cfg         = load_config(config_path)
    paths_cfg   = cfg["paths"]
    ds_cfg      = cfg["dataset"]
    variations  = ds_cfg["variations"]

    clean_base      = paths_cfg["clean_base"]
    final_dataset   = paths_cfg["final_dataset"]
    image_size      = ds_cfg["image_size"]
    max_images      = ds_cfg["max_images"]
    seed            = ds_cfg["seed"]

    # ── 1. Collect source images ───────────────────────────────────────────
    all_images = collect_images(clean_base, max_images)

    # ── 2. Split source images ─────────────────────────────────────────────
    train_imgs, val_imgs, test_imgs = stratified_split(
        all_images,
        ds_cfg["train_split"],
        ds_cfg["val_split"],
        seed,
    )
    logger.info(f"Split: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    # ── 3. Prepare CSV ─────────────────────────────────────────────────────
    os.makedirs(final_dataset, exist_ok=True)
    csv_path = os.path.join(final_dataset, "labels.csv")
    rows = []  # (relative_path, label, split, variation)

    total_clean     = 0
    total_corrupted = 0

    # ── 4. Process each split ──────────────────────────────────────────────
    for split_name, img_list in splits.items():
        logger.info(f"\nProcessing split: {split_name} ({len(img_list)} images)")

        for img_path in tqdm(img_list, desc=f"  Clean → {split_name}"):
            fname = Path(img_path).stem + ".png"
            dest  = os.path.join(final_dataset, split_name, "clean", fname)
            ok = process_and_save(img_path, dest, image_size, variation_cfg=None, seed=seed)
            if ok:
                rel = os.path.relpath(dest, final_dataset)
                rows.append({"path": rel, "label": 0, "split": split_name, "variation": "clean"})
                total_clean += 1

        # ── 5. Generate one corrupted copy per variation ───────────────────
        for var_name, var_cfg in variations.items():
            for i, img_path in enumerate(tqdm(img_list, desc=f"  {var_name} → {split_name}")):
                fname = Path(img_path).stem + f"_{var_name}.png"
                dest  = os.path.join(final_dataset, split_name, var_name, fname)
                ok = process_and_save(
                    img_path, dest, image_size,
                    variation_cfg=var_cfg,
                    seed=seed + i,
                )
                if ok:
                    rel = os.path.relpath(dest, final_dataset)
                    rows.append({"path": rel, "label": 1, "split": split_name, "variation": var_name})
                    total_corrupted += 1

    # ── 6. Write labels.csv ────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "variation"])
        writer.writeheader()
        writer.writerows(rows)

    # ── 7. Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"  Clean images    : {total_clean}")
    logger.info(f"  Corrupted images: {total_corrupted}")
    logger.info(f"  Total           : {total_clean + total_corrupted}")
    logger.info(f"  Labels CSV      : {csv_path}")
    logger.info(f"  Class balance   : {total_clean / (total_clean + total_corrupted):.1%} clean")
    logger.info("="*60)

    # Warn if badly imbalanced (shouldn't happen but good to check)
    ratio = total_clean / max(1, total_corrupted)
    if ratio < 0.14 or ratio > 0.87:
        logger.warning(
            f"Class imbalance detected (clean/corrupted={ratio:.2f}). "
            "Consider adjusting max_images or number of variations."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_dataset(args.config)
