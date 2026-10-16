"""
verify_pairing_visual.py
---------------------------
Content-based pairing check - catches a WRONG pairing even when
recover_pairing.py's metadata validation passes. That metadata check only
confirms the random-number SEQUENCE stayed in sync; it says nothing about
whether clean_base actually has the same files in the same order as the
original generation run (exactly the bug that slipped through last time,
caused by a stray .ipynb_checkpoints file).

This script instead compares ACTUAL PIXELS: since an attacked image is the
clean source + a stripe overlay, the background OUTSIDE the beam-affected
region should still closely resemble the claimed clean source. If pairing
is wrong, the whole scene will differ, not just the stripe area.

Usage:
    python verify_pairing_visual.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --n_samples 30 \
        --seed 42
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from rolling_shutter_simulator_v7 import CameraParams, LaserParams, RollingShutterSimulator
from dataset_builder import MODULATION_MAP


def build_attack_mask(row, cam: CameraParams, sim: RollingShutterSimulator, threshold=0.3):
    """Recompute the spatial attack mask from this row's own stored params -
    same beam model used to generate the image - and threshold it to a
    binary 'attack-affected region'. (Inlined here, same as in
    attack_effect_analysis.py, so this script has no dependency on that file.)"""
    laser = LaserParams(
        frequency=float(row["frequency"]),
        wavelength=int(row["wavelength"]),
        power_mw=float(row["power_mw"]),
        duty_cycle=float(row.get("duty_cycle", 0.5)) if "duty_cycle" in row and not pd.isna(row.get("duty_cycle", None)) else 0.5,
        modulation=MODULATION_MAP.get(row.get("modulation", "square"), list(MODULATION_MAP.values())[0]),
        angle_deg=float(row.get("angle_deg", 0.0)),
        coverage=float(row["coverage"]),
        distance_m=float(row.get("distance_m", 10.0)) if "distance_m" in row and not pd.isna(row.get("distance_m", None)) else 10.0,
        ellipticity=float(row.get("ellipticity", 1.0)) if "ellipticity" in row and not pd.isna(row.get("ellipticity", None)) else 1.0,
    )
    mask, _, _ = sim._coverage_mask(laser)
    return mask >= threshold


def similarity_outside_mask(clean_img, attacked_img, mask):
    """Normalized cross-correlation on the region OUTSIDE the attack mask,
    on grayscale. High (close to 1.0) = same scene. Low/negative = different
    scene entirely - a strong signal of wrong pairing."""
    outside = ~mask
    if outside.sum() < 50:  # mask covers nearly the whole frame, not useful here
        return None

    c_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a_gray = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    c_vals = c_gray[outside]
    a_vals = a_gray[outside]

    c_vals = c_vals - c_vals.mean()
    a_vals = a_vals - a_vals.mean()
    denom = (np.linalg.norm(c_vals) * np.linalg.norm(a_vals))
    if denom < 1e-6:
        return None
    return float(np.dot(c_vals, a_vals) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True, help="labels_paired.csv from recover_pairing.py")
    ap.add_argument("--n_samples", type=int, default=30, help="total samples to check, spread across variations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_examples", default=None,
                     help="optional dir to save a few side-by-side images for manual eyeballing")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]
    final_dataset = project_root / cfg["paths"]["final_dataset"]
    cam = CameraParams(height=224, width=224)
    sim = RollingShutterSimulator(cam, seed=0)
    target_size = (cam.width, cam.height)

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        df = df[df["pairing_verified"] != False]
    df = df[df["variation"] != "clean"].dropna(subset=["source_clean_image"])

    per_var_n = max(1, args.n_samples // df["variation"].nunique())
    sample = df.groupby("variation", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_var_n), random_state=args.seed)
    )

    if args.save_examples:
        out_dir = Path(args.save_examples)
        out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    print(f"Checking {len(sample)} samples...\n")
    for _, row in sample.iterrows():
        src_path = clean_base / row["source_clean_image"]
        attacked_path = final_dataset / str(row["path"]).replace("\\", "/")

        clean_img = cv2.imread(str(src_path))
        attacked_img = cv2.imread(str(attacked_path))
        if clean_img is None or attacked_img is None:
            print(f"[MISSING FILE] {row['path']}")
            continue
        if clean_img.shape[:2] != (target_size[1], target_size[0]):
            clean_img = cv2.resize(clean_img, target_size)

        mask = build_attack_mask(row, cam, sim, threshold=0.3)
        score = similarity_outside_mask(clean_img, attacked_img, mask)
        if score is None:
            continue

        scores.append({"variation": row["variation"], "path": row["path"],
                        "source": row["source_clean_image"], "similarity": score})

        if args.save_examples:
            h = min(clean_img.shape[0], attacked_img.shape[0])
            side_by_side = np.hstack([clean_img, attacked_img])
            name = f"{row['variation']}_{Path(row['path']).stem}_sim{score:.2f}.png"
            cv2.imwrite(str(out_dir / name), side_by_side)

    scores_df = pd.DataFrame(scores)
    print(scores_df.groupby("variation")["similarity"].agg(["mean", "min", "max", "count"]).to_string())

    low_sim = scores_df[scores_df["similarity"] < 0.5]
    print(f"\n--- Verdict ---")
    print(f"Mean similarity overall: {scores_df['similarity'].mean():.3f}")
    print(f"Samples with similarity < 0.5 (likely WRONG pairing): {len(low_sim)} / {len(scores_df)}")

    if scores_df["similarity"].mean() > 0.7 and len(low_sim) == 0:
        print("PASSED - attacked images genuinely resemble their claimed clean source outside "
              "the attack region. Pairing is trustworthy, safe to proceed to object detection.")
    elif len(low_sim) > 0:
        print("WARNING - some samples show low similarity to their claimed source. "
              "This is the exact failure mode that slipped past metadata-only validation "
              "last time (e.g. a stray .ipynb_checkpoints file shifting clean_base's sort "
              "order). Check clean_base's contents before trusting this pairing.")
        print(low_sim.to_string(index=False))
    else:
        print("Inconclusive - low overall similarity but no samples below the hard threshold. "
              "Review manually before proceeding.")


if __name__ == "__main__":
    main()