"""
attack_effect_analysis_full.py
---------------------------------
Full-dataset version of attack_effect_analysis.py. Differences from the
sample-dataset version:
  1. Uses the RAW file in clean_base/ directly as the baseline for each
     attacked row's source_clean_image, instead of requiring a matching
     'clean' row in the same labels file (which isn't guaranteed to exist
     at full-dataset scale, since 'clean' rows sample independently).
  2. CACHES baseline detections per unique source_clean_image - since many
     attacked rows share the same source image (sampled with replacement
     from a smaller pool), this avoids redundant YOLO calls and makes
     full-dataset runs tractable.
  3. Reads labels_paired.csv (from recover_pairing.py) and filters to
     pairing_verified == True automatically.

Run from src/dataset/:

    python attack_effect_analysis_full.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --split test \
        --out_dir ../../data/final_dataset/attack_effect_full_results \
        --model yolov8n.pt \
        --conf 0.25 \
        --iou_thresh 0.3
"""

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
import yaml
from ultralytics import YOLO


def get_detections_resized(model, img_path, target_size, conf):
    """Loads an image and resizes it to match the dataset's actual resolution
    (target_size = (width, height)) BEFORE running detection - critical so
    baseline box coordinates are in the same coordinate space as the
    attacked images, which were generated at this resolution. Running
    detection on the raw original-resolution clean_base file instead would
    put baseline boxes on a totally different scale, making IoU matching
    against the attacked image meaningless."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    if img.shape[:2] != (target_size[1], target_size[0]):
        img = cv2.resize(img, target_size)
    result = model(img, conf=conf, verbose=False)[0]
    boxes = result.boxes
    dets = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].tolist()
        dets.append({
            "box": xyxy,
            "cls": int(boxes.cls[i].item()),
            "cls_name": result.names[int(boxes.cls[i].item())],
            "conf": float(boxes.conf[i].item()),
        })
    return dets

from rolling_shutter_simulator_v7 import CameraParams, LaserParams, RollingShutterSimulator
from dataset_builder import MODULATION_MAP

# reuse the exact same matching/mask logic from the sample-dataset script
from attack_effect_analysis import (
    get_detections, match_detections, build_attack_mask, box_overlap_with_mask,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True, help="labels_paired.csv from recover_pairing.py")
    ap.add_argument("--split", default=None, help="filter to a split (train/val/test), or omit for all")
    ap.add_argument("--out_dir", default="attack_effect_full_results")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--mask_threshold", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        before = len(df)
        df = df[df["pairing_verified"] != False]  # keep True and None(clean rows)
        print(f"Filtered {before - len(df)} unverified rows out ({before} -> {len(df)})")

    if args.split:
        df = df[df["split"] == args.split]
        print(f"Filtered to split={args.split}: {len(df)} rows")

    attacked_df = df[df["variation"] != "clean"]
    print(f"Processing {len(attacked_df)} attacked rows...")

    model = YOLO(args.model)
    cam = CameraParams(height=224, width=224)
    sim = RollingShutterSimulator(cam, seed=0)

    # cache baseline detections per unique source image - avoids redundant inference
    baseline_cache = {}
    unique_sources = attacked_df["source_clean_image"].dropna().unique()
    print(f"Running baseline detection on {len(unique_sources)} unique source images...")
    target_size = (cam.width, cam.height)  # (w, h) - must match how attacked images were generated
    for i, src_name in enumerate(unique_sources):
        src_path = clean_base / src_name
        if not src_path.is_file():
            continue
        baseline_cache[src_name] = get_detections_resized(model, src_path, target_size, args.conf)
        if i % 200 == 0:
            print(f"  baseline {i}/{len(unique_sources)}")

    box_rows = []
    n_processed = 0
    for _, row in attacked_df.iterrows():
        src_name = row.get("source_clean_image")
        if pd.isna(src_name) or src_name not in baseline_cache:
            continue
        clean_dets = baseline_cache[src_name]

        attacked_path = project_root / cfg["paths"]["final_dataset"] / str(row["path"]).replace("\\", "/")
        if not attacked_path.is_file():
            continue

        attacked_dets = get_detections(model, str(attacked_path), args.conf)
        matched, lost, new = match_detections(clean_dets, attacked_dets, args.iou_thresh)
        mask = build_attack_mask(row, cam, sim, threshold=args.mask_threshold)

        for c, a, iou in matched:
            box_rows.append({
                "image_id": row["path"], "variation": row["variation"], "outcome": "matched", "class": c["cls_name"],
                "clean_conf": c["conf"], "attacked_conf": a["conf"], "conf_drop": c["conf"] - a["conf"],
                "iou": iou, "attack_region_overlap": box_overlap_with_mask(c["box"], mask),
                "power_mw": row.get("power_mw"), "coverage": row.get("coverage"),
            })
        for c in lost:
            box_rows.append({
                "image_id": row["path"], "variation": row["variation"], "outcome": "lost", "class": c["cls_name"],
                "clean_conf": c["conf"], "attacked_conf": 0.0, "conf_drop": c["conf"],
                "iou": 0.0, "attack_region_overlap": box_overlap_with_mask(c["box"], mask),
                "power_mw": row.get("power_mw"), "coverage": row.get("coverage"),
            })
        for a in new:
            box_rows.append({
                "image_id": row["path"], "variation": row["variation"], "outcome": "new_false_positive", "class": a["cls_name"],
                "clean_conf": 0.0, "attacked_conf": a["conf"], "conf_drop": -a["conf"],
                "iou": 0.0, "attack_region_overlap": box_overlap_with_mask(a["box"], mask),
                "power_mw": row.get("power_mw"), "coverage": row.get("coverage"),
            })

        n_processed += 1
        if n_processed % 500 == 0:
            print(f"  processed {n_processed}/{len(attacked_df)} attacked rows")

    box_df = pd.DataFrame(box_rows)
    box_csv = os.path.join(args.out_dir, "box_level_results_full.csv")
    box_df.to_csv(box_csv, index=False)
    print(f"\nBox-level results saved to {box_csv}")

    summary_rows = []
    for var, g in box_df.groupby("variation"):
        n_clean_boxes = len(g[g["outcome"].isin(["matched", "lost"])])
        n_matched = len(g[g["outcome"] == "matched"])
        n_lost = len(g[g["outcome"] == "lost"])
        n_new = len(g[g["outcome"] == "new_false_positive"])
        lost_in_region = g[(g["outcome"] == "lost") & (g["attack_region_overlap"] >= 0.3)]
        mean_conf_drop_matched = g[g["outcome"] == "matched"]["conf_drop"].mean() if n_matched else float("nan")

        summary_rows.append({
            "variation": var, "n_clean_boxes": n_clean_boxes, "n_matched": n_matched, "n_lost": n_lost,
            "pct_lost": (n_lost / n_clean_boxes * 100) if n_clean_boxes else float("nan"),
            "n_lost_in_attack_region": len(lost_in_region),
            "pct_lost_explained_by_attack_region": (len(lost_in_region) / n_lost * 100) if n_lost else float("nan"),
            "n_new_false_positives": n_new,
            "mean_confidence_drop_matched_boxes": mean_conf_drop_matched,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("pct_lost", ascending=False)
    summary_csv = os.path.join(args.out_dir, "attack_effect_summary_full.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
