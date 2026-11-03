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
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights



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


FEATURE_COLS = ["num_detections", "mean_confidence", "max_confidence", "std_confidence", "low_conf_count"]
OUT_COLS = ["path", "variation", "source_clean_image", "is_attacked"] + FEATURE_COLS


def load_existing(out_path):
    """Rows already saved by an earlier (possibly killed) run. A session killed mid-write can leave a
    truncated last line; such rows are dropped and the file is rewritten clean so they get redone."""
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.read_csv(out_path, on_bad_lines="skip")
    n0 = len(df)
    df = df.dropna(subset=["path", "variation", "is_attacked"] + FEATURE_COLS)
    df = df.drop_duplicates(subset="path", keep="last")
    if len(df) != n0:
        df.to_csv(out_path, index=False)
        print(f"[resume] removed {n0 - len(df)} incomplete/duplicate rows from {out_path}")
    return df


def append_rows(out_path, rows):
    """Append a chunk to the output CSV and force it to disk, so a killed session loses at most one chunk."""
    if not rows:
        return
    new = pd.DataFrame(rows)[OUT_COLS]
    exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    if exists:                                   # keep the column order of the file already on disk
        new = new[pd.read_csv(out_path, nrows=0).columns.tolist()]
    with open(out_path, "a", newline="") as f:
        new.to_csv(f, header=not exists, index=False)
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max_per_variation", type=int, default=None,
                    help="randomly keep at most N images per ATTACK variation (the full dataset is "
                         "~97k images; ~1 hour of SSD inference). Omit to use everything.")
    ap.add_argument("--max_clean", type=int, default=None,
                    help="randomly keep at most M clean images. Default (when --max_per_variation is "
                         "set): as many clean images as total attacked images kept, so the mix is "
                         "~50/50. The full dataset is ~87%% attacked, which makes plain accuracy "
                         "misleading (always answering 'attacked' scores 87%%).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new", type=int, default=None,
                    help="process at most N NEW images in this run, then stop (e.g. 1000 per Kaggle session). "
                         "Re-running the same command continues where the output file left off.")
    ap.add_argument("--flush_every", type=int, default=200,
                    help="append to the output file every N images (a killed session loses at most this many)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    final_dataset = project_root / cfg["paths"]["final_dataset"]
    cam_cfg = cfg.get("camera", {}) or {}
    target_size = (int(cam_cfg.get("width", 224)), int(cam_cfg.get("height", 224)))   # (w, h)

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        df = df[df["pairing_verified"] != False]
    if args.split:
        df = df[df["split"] == args.split]

    if args.max_per_variation is not None:
        att = df[df["variation"] != "clean"]
        # plain comprehension instead of groupby.apply: apply() drops the grouping column on
        # pandas 3.x but keeps it on 2.x, which would silently lose 'variation' on one of them
        att = pd.concat([g.sample(min(len(g), args.max_per_variation), random_state=args.seed)
                         for _, g in att.groupby("variation")])
        clean = df[df["variation"] == "clean"]
        n_clean = args.max_clean if args.max_clean is not None else len(att)
        clean = clean.sample(min(len(clean), n_clean), random_state=args.seed)
        df = pd.concat([att, clean]).sample(frac=1.0, random_state=args.seed)
        print(f"Subsampled to {len(att)} attacked + {len(clean)} clean = {len(df)} images")
    elif args.max_clean is not None:
        clean = df[df["variation"] == "clean"].sample(
            min((df["variation"] == "clean").sum(), args.max_clean), random_state=args.seed)
        df = pd.concat([df[df["variation"] != "clean"], clean])

    # Fixed shuffle: any prefix of the output is a random, representative sample, so even a partly
    # finished extraction is usable for training.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    done = load_existing(args.out)
    todo = df[~df["path"].isin(set(done["path"]))]
    print(f"{len(done)} rows already in {args.out}; {len(todo)} images remaining")
    if args.max_new is not None:
        todo = todo.head(args.max_new)
        print(f"processing {len(todo)} this run (--max_new {args.max_new})")
    if len(todo) == 0:
        print("nothing to do - extraction is complete for these settings")
    else:
        model, preprocess = load_model(device)
        buffer, n_new, n_skipped = [], 0, 0
        try:
            for n_done, (_, row) in enumerate(todo.iterrows()):
                img_path = final_dataset / str(row["path"]).replace("\\", "/")
                img = cv2.imread(str(img_path)) if img_path.is_file() else None
                if img is None:
                    n_skipped += 1
                    continue
                if img.shape[:2] != (target_size[1], target_size[0]):
                    img = cv2.resize(img, target_size)

                feats = extract_features(model, preprocess, img, args.conf, device)
                feats["path"] = row["path"]
                feats["variation"] = row["variation"]
                feats["source_clean_image"] = row.get("source_clean_image")   # used to split by background
                feats["is_attacked"] = int(row["variation"] != "clean")       # ground-truth binary label
                buffer.append(feats)
                n_new += 1

                if len(buffer) >= args.flush_every:
                    append_rows(args.out, buffer)
                    buffer = []
                if n_done % 500 == 0:
                    print(f"  {n_done}/{len(todo)} processed this run", flush=True)
        finally:
            append_rows(args.out, buffer)          # also runs on Ctrl-C / exceptions
        print(f"\nthis run: {n_new} new rows saved" + (f", {n_skipped} images missing/unreadable" if n_skipped else ""))

    final = load_existing(args.out)
    print(f"{args.out} now has {len(final)} rows")
    if len(final):
        print(final.groupby("variation")[["num_detections", "mean_confidence"]].mean().to_string())

if __name__ == "__main__":
    main()
