"""
audit_pixels.py
----------------
Verifies stripe visibility by reading actual pixel data from a sample of
generated images - independent of labels.csv metadata entirely. Use this
if you're unsure whether existing images are correct without regenerating
them (e.g. after discovering a metadata-only CSV bug that doesn't affect
the images themselves).

Method: for each sampled image, compute the row-mean brightness profile
(average pixel intensity per row), then high-pass filter it (subtract a
smoothed version) to separate the fast row-to-row banding signal from slow
scene gradients (sky, road, buildings). Reports the std of that residual
per variation, and compares attacked variations against the clean baseline
- attacked images should show a clearly higher residual std if stripes are
actually present in the pixels.

Usage:
    python audit_pixels.py --dataset data/final_dataset --samples 15
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def row_profile_residual_std(img_path: str, smooth_windows=(9, 41)) -> float:
    """Returns the MAX residual std across multiple smoothing scales - a
    single small window under-detects broad/low-frequency bands (they get
    partially absorbed into the "scene" estimate instead of showing up as
    residual), while a single large window can over-smooth genuinely fine
    high-frequency stripes. Using both scales and taking the max catches
    stripes across the whole freq_low_wide -> freq_ultra_aliasing range."""
    img = cv2.imread(img_path)
    if img is None:
        return float("nan")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    row_profile = gray.mean(axis=1)  # mean brightness per row

    best = 0.0
    for window in smooth_windows:
        kernel = np.ones(window) / window
        padded = np.pad(row_profile, window // 2, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[:len(row_profile)]
        residual_std = float((row_profile - smoothed).std())
        best = max(best, residual_std)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final_dataset")
    ap.add_argument("--samples", type=int, default=15, help="images to sample per variation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    labels_path = dataset_root / "labels.csv"

    by_variation = defaultdict(list)
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_variation[row["variation"]].append(row["path"])

    rng = random.Random(args.seed)
    results = {}
    for variation, paths in sorted(by_variation.items()):
        sample = rng.sample(paths, min(args.samples, len(paths)))
        stds = [row_profile_residual_std(str(dataset_root / p)) for p in sample]
        stds = [s for s in stds if not np.isnan(s)]
        results[variation] = (np.mean(stds), np.std(stds), len(stds))

    print(f"{'variation':<22} {'n':>4} {'mean_residual_std':>18} {'std_of_std':>12}")
    print("-" * 60)
    clean_mean = results.get("clean", (None,))[0]
    for variation, (mean_std, std_std, n) in results.items():
        flag = ""
        if variation != "clean" and clean_mean is not None:
            ratio = mean_std / clean_mean if clean_mean > 0 else float("inf")
            flag = f"  ({ratio:.1f}x clean baseline)"
        print(f"{variation:<22} {n:>4} {mean_std:>18.3f} {std_std:>12.3f}{flag}")

    print("\nInterpretation: attacked variations should show a residual std "
          "clearly higher than 'clean' (roughly 2x or more is a good sign; "
          "close to 1x means the attack isn't visually distinguishable in "
          "actual pixel content, regardless of what any metadata claims).")


if __name__ == "__main__":
    main()
