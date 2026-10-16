"""
extract_detection_features.py
--------------------------------
Runs SSD300 on every frame (clean + attacked) and extracts per-image
detection-based features. This is the DETECTION-SIDE half of the
CNN+detector fusion classifier - the other half (CNN class probabilities)
needs your trained CNN's inference code, which this script does not
depend on, so you can run this part immediately.

Features extracted per image (no temporal/sequence assumption - these are
independent frames, not video):
  - num_detections       : raw detection count at the operating threshold
  - mean_confidence       : mean confidence across all detections (0 if none)
  - max_confidence        : highest single detection confidence (0 if none)
  - std_confidence         : spread of detection confidences (0 if <2 dets)
  - low_conf_count         : detections just above threshold (conf < 0.4) -
                              a frame with many borderline detections is a
                              different signal than one with a few confident
                              ones, even at the same total count

These features are label-agnostic at this stage (no ground truth used) -
this is pure signal extraction, meant to be joined later with your CNN's
output and the true clean/attacked label for training the fusion model.

Usage:
    python extract_detection_features.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --split test \
        --out ../../data/final_dataset/detection_features_ssd.csv \
        --conf 0.15
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from rolling_shutter_simulator_v7 import CameraParams


def load_model(device):
    weights = SSD300_VGG16_Weights.COCO_V1
    model = ssd300_vgg16(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess


def extract_features(model, preprocess, img_bgr, conf_floor, device):
    """conf_floor is intentionally lower than a normal detection threshold
    (e.g. 0.15 instead of 0.25) - for FEATURE EXTRACTION we want to capture
    weak/borderline detections too, since their presence/absence is itself
    informative signal for the fusion classifier, not just confident ones."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = preprocess(tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)[0]

    scores = output["scores"].cpu().numpy()
    scores = scores[scores >= conf_floor]

    if len(scores) == 0:
        return {
            "num_detections": 0, "mean_confidence": 0.0, "max_confidence": 0.0,
            "std_confidence": 0.0, "low_conf_count": 0,
        }
    return {
        "num_detections": len(scores),
        "mean_confidence": float(scores.mean()),
        "max_confidence": float(scores.max()),
        "std_confidence": float(scores.std()) if len(scores) > 1 else 0.0,
        "low_conf_count": int(((scores >= conf_floor) & (scores < 0.4)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    final_dataset = project_root / cfg["paths"]["final_dataset"]
    cam = CameraParams(**cfg.get("camera", {}))
    target_size = (cam.width, cam.height)

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        df = df[df["pairing_verified"] != False]
    if args.split:
        df = df[df["split"] == args.split]

    model, preprocess = load_model(device)

    rows = []
    for i, row in df.iterrows():
        img_path = final_dataset / str(row["path"]).replace("\\", "/")
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (target_size[1], target_size[0]):
            img = cv2.resize(img, target_size)

        feats = extract_features(model, preprocess, img, args.conf, device)
        feats["path"] = row["path"]
        feats["variation"] = row["variation"]
        feats["is_attacked"] = int(row["variation"] != "clean")  # ground-truth binary label
        rows.append(feats)

        if i % 500 == 0:
            print(f"  {i}/{len(df)} processed")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.out}")
    print(out_df.groupby("variation")[["num_detections", "mean_confidence"]].mean().to_string())


if __name__ == "__main__":
    main()
