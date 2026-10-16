"""
run_ssd_detection.py
-----------------------
Same box-level attack-effect methodology as attack_effect_analysis_full.py,
but using torchvision's SSD300 (COCO-pretrained) instead of YOLOv8 - the
SAME detector family Kohler et al. tested against, letting you validate
whether your synthetic simulator reproduces a comparable real-world effect
on the same architecture they used.

Reuses the matching/mask logic from attack_effect_analysis.py (unchanged) -
only the detection function is different (torchvision API instead of
ultralytics).

Run from src/dataset/:

    python run_ssd_detection.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --split test \
        --out_dir ../../data/final_dataset/attack_effect_full_results_ssd \
        --conf 0.25 \
        --iou_thresh 0.3
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from rolling_shutter_simulator_v7 import CameraParams, LaserParams, RollingShutterSimulator
from dataset_builder import MODULATION_MAP
from attack_effect_analysis import match_detections, build_attack_mask, box_overlap_with_mask


def load_model(device):
    weights = SSD300_VGG16_Weights.COCO_V1
    model = ssd300_vgg16(weights=weights)
    model.eval().to(device)
    categories = weights.meta["categories"]
    preprocess = weights.transforms()
    return model, categories, preprocess


def get_detections_ssd(model, categories, preprocess, img_bgr, conf, device):
    """img_bgr: np.array (H,W,3) in BGR (cv2 convention). Returns list of dets
    in the same dict format as the YOLO pipeline, so downstream matching code
    is unchanged."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = preprocess(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)[0]

    dets = []
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score < conf:
            continue
        dets.append({
            "box": box.tolist(),
            "cls": int(label),
            "cls_name": categories[int(label)],
            "conf": float(score),
        })
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default="attack_effect_full_results_ssd")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--mask_threshold", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]
    final_dataset = project_root / cfg["paths"]["final_dataset"]

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        before = len(df)
        df = df[df["pairing_verified"] != False]
        print(f"Filtered {before - len(df)} unverified rows out")
    if args.split:
        df = df[df["split"] == args.split]
        print(f"Filtered to split={args.split}: {len(df)} rows")

    attacked_df = df[df["variation"] != "clean"].dropna(subset=["source_clean_image"])
    print(f"Processing {len(attacked_df)} attacked rows with SSD300...")

    model, categories, preprocess = load_model(device)
    cam = CameraParams(height=224, width=224)
    sim = RollingShutterSimulator(cam, seed=0)
    target_size = (cam.width, cam.height)

    baseline_cache = {}
    unique_sources = attacked_df["source_clean_image"].unique()
    print(f"Running baseline SSD detection on {len(unique_sources)} unique source images...")
    for i, src_name in enumerate(unique_sources):
        src_path = clean_base / src_name
        img = cv2.imread(str(src_path))
        if img is None:
            continue
        if img.shape[:2] != (target_size[1], target_size[0]):
            img = cv2.resize(img, target_size)
        baseline_cache[src_name] = get_detections_ssd(model, categories, preprocess, img, args.conf, device)
        if i % 200 == 0:
            print(f"  baseline {i}/{len(unique_sources)}")

    box_rows = []
    n_processed = 0
    for _, row in attacked_df.iterrows():
        src_name = row["source_clean_image"]
        if src_name not in baseline_cache:
            continue
        clean_dets = baseline_cache[src_name]

        attacked_path = final_dataset / str(row["path"]).replace("\\", "/")
        if not attacked_path.is_file():
            continue
        attacked_img = cv2.imread(str(attacked_path))
        if attacked_img is None:
            continue
        attacked_dets = get_detections_ssd(model, categories, preprocess, attacked_img, args.conf, device)

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
            print(f"  processed {n_processed}/{len(attacked_df)}")

    box_df = pd.DataFrame(box_rows)
    box_csv = os.path.join(args.out_dir, "box_level_results_ssd.csv")
    box_df.to_csv(box_csv, index=False)
    print(f"\nBox-level results saved to {box_csv}")
    print("\nRun summarize_results.py on this file to get the percentage table:")
    print(f"  python summarize_results.py --box_csv {box_csv} --out_dir {args.out_dir} --label ssd300")


if __name__ == "__main__":
    main()
