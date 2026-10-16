"""
attack_effect_analysis.py
----------------------------
Same methodology as Kohler et al.'s compute_attack_effect (from their
attack_object_detection.py) - box-level comparison between baseline (clean)
and attacked frame detections, checking whether lost detections are
spatially correlated with the attack-affected region - but built on YOLOv8
(works today, no TF1/BDD100K dependency) and your own dataset.

Requires the PAIRED sample dataset (labels_sample.csv with
source_clean_image column) from generate_sample_dataset.py, since this
needs true same-scene clean/attacked pairs to do box-level matching -
population-level comparison (run_object_detection.py) can't do this.

For each source_clean_image:
  1. Run YOLO on the clean version -> baseline boxes
  2. Run YOLO on each attacked variation of the SAME scene -> attacked boxes
  3. Match boxes clean<->attacked by IoU + class (like Kohler's approach)
     -> matched  = detection survived the attack
     -> lost     = present in clean, gone in attacked
     -> new      = present in attacked, not in clean (false positive
                    introduced by the attack, e.g. a stripe misread as an
                    object)
  4. For each LOST box, recompute the attack's spatial coverage mask from
     the row's own stored laser parameters (frequency, wavelength, power_mw,
     coverage, angle_deg, distance_m, ellipticity) using the real simulator's
     _coverage_mask() - the same beam mask used to generate the image - and
     check what fraction of that box overlaps the attack-affected region.
     This is the causal check: "did this object get lost BECAUSE the stripe
     was over it" vs "did it get lost for some other reason."

Run from src/dataset/ (needs rolling_shutter_simulator_v7.py and
dataset_builder.py importable):

    python attack_effect_analysis.py \
        --labels ../../data/sample_dataset/labels_sample.csv \
        --data_root ../../data/sample_dataset \
        --out_dir ../../data/sample_dataset/attack_effect_results \
        --ae_mode ae_on \
        --model yolov8n.pt \
        --conf 0.25 \
        --iou_thresh 0.3
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from rolling_shutter_simulator_v7 import CameraParams, LaserParams, RollingShutterSimulator
from dataset_builder import MODULATION_MAP


def box_iou(box_a, box_b):
    """boxes as [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_detections(model, img_path, conf):
    result = model(img_path, conf=conf, verbose=False)[0]
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


def match_detections(clean_dets, attacked_dets, iou_thresh):
    """Greedy IoU+class matching. Returns (matched_pairs, lost, new)."""
    matched_pairs = []
    used_attacked = set()
    for c in clean_dets:
        best_iou, best_j = 0.0, None
        for j, a in enumerate(attacked_dets):
            if j in used_attacked or a["cls"] != c["cls"]:
                continue
            iou = box_iou(c["box"], a["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None and best_iou >= iou_thresh:
            matched_pairs.append((c, attacked_dets[best_j], best_iou))
            used_attacked.add(best_j)
        else:
            matched_pairs.append((c, None, 0.0))  # lost

    lost = [c for c, a, _ in matched_pairs if a is None]
    matched = [(c, a, iou) for c, a, iou in matched_pairs if a is not None]
    new = [a for j, a in enumerate(attacked_dets) if j not in used_attacked]
    return matched, lost, new


def build_attack_mask(row, cam: CameraParams, sim: RollingShutterSimulator, threshold=0.3):
    """Recompute the spatial attack mask from this row's own stored params -
    same beam model used to generate the image - and threshold it to a
    binary 'attack-affected region'."""
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


def box_overlap_with_mask(box, mask):
    h, w = mask.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    return float(region.mean()) if region.size > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="attack_effect_results")
    ap.add_argument("--ae_mode", default="ae_on")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--mask_threshold", type=float, default=0.3,
                     help="beam intensity threshold to count as 'attack-affected region'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.labels)
    if "ae_mode" in df.columns:
        df = df[df["ae_mode"] == args.ae_mode]
    if "source_clean_image" not in df.columns:
        raise ValueError("labels file has no source_clean_image column - "
                          "this script needs the paired sample dataset from generate_sample_dataset.py")

    model = YOLO(args.model)
    cam = CameraParams(height=224, width=224)  # matches project's camera config
    sim = RollingShutterSimulator(cam, seed=0)

    box_rows = []
    summary_rows = []

    scenes = df["source_clean_image"].unique()
    for scene in scenes:
        scene_df = df[df["source_clean_image"] == scene]
        clean_row = scene_df[scene_df["variation"] == "clean"]
        if clean_row.empty:
            continue
        clean_path = os.path.join(args.data_root, clean_row.iloc[0]["path"])
        clean_dets = get_detections(model, clean_path, args.conf)

        for _, row in scene_df[scene_df["variation"] != "clean"].iterrows():
            attacked_path = os.path.join(args.data_root, row["path"])
            attacked_dets = get_detections(model, attacked_path, args.conf)

            matched, lost, new = match_detections(clean_dets, attacked_dets, args.iou_thresh)
            mask = build_attack_mask(row, cam, sim, threshold=args.mask_threshold)

            for c, a, iou in matched:
                box_rows.append({
                    "scene": scene, "variation": row["variation"], "outcome": "matched",
                    "class": c["cls_name"], "clean_conf": c["conf"], "attacked_conf": a["conf"],
                    "conf_drop": c["conf"] - a["conf"], "iou": iou,
                    "attack_region_overlap": box_overlap_with_mask(c["box"], mask),
                })
            for c in lost:
                box_rows.append({
                    "scene": scene, "variation": row["variation"], "outcome": "lost",
                    "class": c["cls_name"], "clean_conf": c["conf"], "attacked_conf": 0.0,
                    "conf_drop": c["conf"], "iou": 0.0,
                    "attack_region_overlap": box_overlap_with_mask(c["box"], mask),
                })
            for a in new:
                box_rows.append({
                    "scene": scene, "variation": row["variation"], "outcome": "new_false_positive",
                    "class": a["cls_name"], "clean_conf": 0.0, "attacked_conf": a["conf"],
                    "conf_drop": -a["conf"], "iou": 0.0,
                    "attack_region_overlap": box_overlap_with_mask(a["box"], mask),
                })

    box_df = pd.DataFrame(box_rows)
    box_csv = os.path.join(args.out_dir, "box_level_results.csv")
    box_df.to_csv(box_csv, index=False)
    print(f"Box-level results saved to {box_csv}")

    # summary per variation
    for var, g in box_df.groupby("variation"):
        n_clean_boxes = len(g[g["outcome"].isin(["matched", "lost"])])
        n_matched = len(g[g["outcome"] == "matched"])
        n_lost = len(g[g["outcome"] == "lost"])
        n_new = len(g[g["outcome"] == "new_false_positive"])
        lost_in_region = g[(g["outcome"] == "lost") & (g["attack_region_overlap"] >= 0.3)]
        lost_outside_region = g[(g["outcome"] == "lost") & (g["attack_region_overlap"] < 0.3)]
        mean_conf_drop_matched = g[g["outcome"] == "matched"]["conf_drop"].mean() if n_matched else float("nan")

        summary_rows.append({
            "variation": var,
            "n_clean_boxes": n_clean_boxes,
            "n_matched": n_matched,
            "n_lost": n_lost,
            "pct_lost": (n_lost / n_clean_boxes * 100) if n_clean_boxes else float("nan"),
            "n_lost_in_attack_region": len(lost_in_region),
            "n_lost_outside_attack_region": len(lost_outside_region),
            "pct_lost_explained_by_attack_region": (len(lost_in_region) / n_lost * 100) if n_lost else float("nan"),
            "n_new_false_positives": n_new,
            "mean_confidence_drop_matched_boxes": mean_conf_drop_matched,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("pct_lost", ascending=False)
    summary_csv = os.path.join(args.out_dir, "attack_effect_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
