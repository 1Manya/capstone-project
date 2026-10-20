"""
visualize_forced_boxes.py
----------------------------
Shows the SAME box (from the clean detection) on BOTH the clean and
attacked image - so you can see exactly what happened to that specific
object, even if the attacked-frame model's own confidence for it is 0
(completely lost). This is different from result.plot() (YOLO) or drawing
only the model's own attacked-frame detections, which simply show nothing
for a lost object instead of showing "this box existed, and here's what
confidence it has now."

Works with EITHER detector via --detector yolo|ssd, using the same
matching logic as attack_effect_analysis_full.py / run_ssd_detection.py,
so the "attacked confidence" shown is exactly what that pipeline computed.

Run from src/dataset/:

    # YOLO
    python visualize_forced_boxes.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --split test --detector yolo --model yolov8n.pt \
        --out_dir ../../data/final_dataset/attack_effect_full_results/forced_boxes_yolo \
        --n_examples 6 --seed 42

    # SSD
    python visualize_forced_boxes.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --split test --detector ssd \
        --out_dir ../../data/final_dataset/attack_effect_full_results_ssd/forced_boxes_ssd \
        --n_examples 6 --seed 42
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from rolling_shutter_simulator_v7 import CameraParams
from attack_effect_analysis import match_detections


# ---------------- detector backends ----------------

def load_yolo(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)

    def detect(img_bgr, conf):
        result = model(img_bgr, conf=conf, verbose=False)[0]
        boxes = result.boxes
        dets = []
        for i in range(len(boxes)):
            dets.append({
                "box": boxes.xyxy[i].tolist(),
                "cls": int(boxes.cls[i].item()),
                "cls_name": result.names[int(boxes.cls[i].item())],
                "conf": float(boxes.conf[i].item()),
            })
        return dets

    return detect


def load_ssd():
    import torch
    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = SSD300_VGG16_Weights.COCO_V1
    model = ssd300_vgg16(weights=weights)
    model.eval().to(device)
    categories = weights.meta["categories"]
    preprocess = weights.transforms()

    def detect(img_bgr, conf):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = preprocess(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)[0]
        dets = []
        for box, score, label in zip(output["boxes"].cpu().numpy(),
                                      output["scores"].cpu().numpy(),
                                      output["labels"].cpu().numpy()):
            if score < conf:
                continue
            dets.append({"box": box.tolist(), "cls": int(label), "cls_name": categories[int(label)], "conf": float(score)})
        return dets

    return detect


# ---------------- drawing ----------------

def draw_box(img, box, label, color):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_banner(img, text):
    h, w = img.shape[:2]
    banner_h = max(26, h // 10)
    banner = np.full((banner_h, w, 3), 255, dtype=np.uint8)
    cv2.putText(banner, text, (5, banner_h - 6), cv2.FONT_HERSHEY_SIMPLEX, banner_h / 55.0, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--detector", choices=["yolo", "ssd"], required=True)
    ap.add_argument("--model", default="yolov8n.pt", help="only used when --detector yolo")
    ap.add_argument("--out_dir", default="forced_boxes")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--class_agnostic", action="store_true",
                    help="match boxes ignoring class (Köhler et al.'s 'hidden' definition; pair with --iou_thresh 0.5)")
    ap.add_argument("--n_examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]
    final_dataset = project_root / cfg["paths"]["final_dataset"]
    cam = CameraParams(**cfg.get("camera", {}))
    target_size = (cam.width, cam.height)

    detect = load_yolo(args.model) if args.detector == "yolo" else load_ssd()

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        df = df[df["pairing_verified"] != False]
    if args.split:
        df = df[df["split"] == args.split]
    df = df[df["variation"] != "clean"].dropna(subset=["source_clean_image"])

    baseline_cache = {}

    def get_baseline(src_name):
        if src_name not in baseline_cache:
            img = cv2.imread(str(clean_base / src_name))
            if img is None:
                return None, []
            if img.shape[:2] != (target_size[1], target_size[0]):
                img = cv2.resize(img, target_size)
            dets = detect(img, args.conf)
            baseline_cache[src_name] = (img, dets)
        return baseline_cache[src_name]

    for var in df["variation"].unique():
        var_df = df[df["variation"] == var].sample(frac=1.0, random_state=args.seed).to_dict("records")
        saved = 0
        for row in var_df:
            if saved >= args.n_examples:
                break
            src_name = row["source_clean_image"]
            clean_img, clean_dets = get_baseline(src_name)
            if clean_img is None or not clean_dets:
                continue

            attacked_path = final_dataset / str(row["path"]).replace("\\", "/")
            if not attacked_path.is_file():
                continue
            attacked_img = cv2.imread(str(attacked_path))
            if attacked_img is None:
                continue
            attacked_dets = detect(attacked_img, args.conf)

            matched, lost, new = match_detections(clean_dets, attacked_dets, args.iou_thresh, class_agnostic=args.class_agnostic)

            left = clean_img.copy()
            right = attacked_img.copy()

            for c in clean_dets:
                left = draw_box(left, c["box"], f"{c['cls_name']} {c['conf']:.2f}", (0, 200, 0))

            # force-draw the SAME box location from clean onto the attacked
            # image, labeled with whatever confidence it has THERE now -
            # matched -> its real attacked confidence; lost -> 0.00
            for c, a, iou in matched:
                right = draw_box(right, c["box"], f"{c['cls_name']} {a['conf']:.2f}", (0, 200, 0))
            for c in lost:
                right = draw_box(right, c["box"], f"{c['cls_name']} 0.00", (0, 0, 255))
            # attack-introduced false positives, shown separately in a
            # distinct color so they're not confused with real lost objects
            for a in new:
                right = draw_box(right, a["box"], f"FP:{a['cls_name']} {a['conf']:.2f}", (0, 140, 255))

            left = draw_banner(left, f"clean ({len(clean_dets)} det)")
            right = draw_banner(right, f"{var} - forced boxes ({len(lost)} lost, {len(matched)} survived)")

            h = min(left.shape[0], right.shape[0])
            left_r = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
            right_r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
            grid = np.hstack([left_r, right_r])

            out_name = f"{var}_{Path(src_name).stem}.png"
            cv2.imwrite(str(out_dir / out_name), grid)
            saved += 1

        print(f"[{var}] saved {saved} forced-box comparison examples")

    print(f"\nAll examples saved to {out_dir}/")
    print("Legend: green = detected (clean or survived), red = lost (0.00 conf on attacked frame), "
          "orange 'FP:' = false positive introduced by the attack")


if __name__ == "__main__":
    main()
