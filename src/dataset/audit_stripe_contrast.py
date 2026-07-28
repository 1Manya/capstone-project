"""
audit_stripe_contrast.py
-------------------------
Run this against your ALREADY-GENERATED labels.csv to measure how many
attacked images actually have washed-out stripe contrast, using the real
exposure_time that was used at generation time (1ms fixed, since this
predates the per-variation exposure_time fix).

No images are touched or regenerated - this only recomputes the row_power
signal from each row's logged frequency/modulation, which is cheap (no
image I/O), so it runs in well under a minute even on 97,500 rows.

Usage:
    python audit_stripe_contrast.py --labels data/final_dataset/labels.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

FPS = 30.0
HEIGHT = 224
EXPOSURE_TIME_USED = 0.001  # 1ms - what your completed run actually used
LOW_CONTRAST_THRESHOLD = 0.08  # row_power std below this = visually near-invisible


def row_power_std(frequency, modulation, duty_cycle, phase, exposure_time,
                   num_samples=24, seed=0):
    if frequency <= 0:
        return 0.0
    frame_time = 1.0 / FPS
    row_time = frame_time / HEIGHT
    rows = np.arange(HEIGHT)
    t_start = rows * row_time
    offsets = np.linspace(0.0, exposure_time, num_samples)
    T = t_start[:, None] + offsets[None, :]
    period = 1.0 / frequency
    phase_arr = np.mod(T + phase, period)

    if modulation in ("square", "pulse"):
        wave = (phase_arr < duty_cycle * period).astype(np.float32)
    elif modulation == "sine":
        wave = (np.sin(2 * np.pi * phase_arr / period) + 1.0) / 2.0
    elif modulation == "triangle":
        half = period / 2.0
        wave = np.where(phase_arr < half, phase_arr / half, 2.0 - phase_arr / half)
    else:  # flicker - random per row/sample, not phase-coherent
        rng = np.random.default_rng(seed)
        wave = rng.random((HEIGHT, num_samples))

    return float(wave.mean(axis=1).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/final_dataset/labels.csv")
    ap.add_argument("--threshold", type=float, default=LOW_CONTRAST_THRESHOLD)
    args = ap.parse_args()

    per_variation = defaultdict(lambda: {"total": 0, "low_contrast": 0, "stds": []})

    with open(args.labels, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variation = row["variation"]
            if variation == "clean":
                continue
            freq = float(row["frequency"])
            modulation = row["modulation"]
            duty_cycle = float(row["duty_cycle"])
            # phase wasn't logged in the original run - use 0.0 as an approximation;
            # phase shifts WHERE stripes fall, not whether contrast survives, so this
            # doesn't change the std much either way
            std = row_power_std(freq, modulation, duty_cycle, 0.0, EXPOSURE_TIME_USED)
            d = per_variation[variation]
            d["total"] += 1
            d["stds"].append(std)
            if std < args.threshold:
                d["low_contrast"] += 1

    print(f"{'variation':<22} {'total':>7} {'low_contrast':>13} {'pct':>7} {'mean_std':>10}")
    print("-" * 65)
    grand_total = grand_low = 0
    for variation, d in sorted(per_variation.items()):
        pct = 100.0 * d["low_contrast"] / d["total"]
        mean_std = np.mean(d["stds"])
        print(f"{variation:<22} {d['total']:>7} {d['low_contrast']:>13} {pct:>6.1f}% {mean_std:>10.4f}")
        grand_total += d["total"]
        grand_low += d["low_contrast"]

    print("-" * 65)
    print(f"{'TOTAL (attacked)':<22} {grand_total:>7} {grand_low:>13} "
          f"{100.0*grand_low/grand_total:>6.1f}%")
    print(f"\n(threshold: row_power std < {args.threshold} = visually near-invisible stripes)")


if __name__ == "__main__":
    main()
