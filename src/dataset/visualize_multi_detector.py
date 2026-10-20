"""
visualize_multi_detector.py
-----------------------------
Qualitative figure for the paper: the SAME frame, one ROW per detector
(default SSD300, YOLOv8n, YOLOv8m), clean frame on the left, attacked
frame on the right.

Boxes on the attacked frame are the boxes the detector found on the CLEAN
frame, forced onto the attacked image:
  green  - object still detected (label shows its attacked-frame confidence)
  red    - object suppressed (label shows 0.00)
  orange - false positive induced by the attack (not present in clean)
This shows what happened to each specific object even when the detector's
own confidence on the attacked frame is zero.

Frames are drawn at random (seeded) among attacked frames whose clean version
has at least one detection for the FIRST detector. That is an illustrative
selection, not a statistical one - say so in the caption. Optionally restrict
to strong attacks with --min_severity_pct.

Run from src/dataset/:
    python visualize_multi_detector.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --split test --n_examples 3 --seed 42 \
        --out_dir ../../data/final_dataset/qualitative_figures
    # Köhler-style matching:   --iou_thresh 0.5 --class_agnostic
    # custom detector list:    --detectors ssd,yolo:yolov8n.pt,yolo:yolov8m.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GREEN, RED, ORANGE = (0, 170, 0), (0, 0, 230), (0, 140, 255)   # BGR
SCALE = 2          # frames are 224x224: upscale first so text stays sharp


def draw_box(img, box, label, color, scale=SCALE):
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, h), _ = cv2.getTextSize(label, font, fs, th)
    top = max(0, y1 - h - 5)
    cv2.rectangle(img, (x1, top), (x1 + tw + 4, top + h + 5), color, -1)
    cv2.putText(img, label, (x1 + 2, top + h + 1), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def banner(img, text, height=26):
    b = np.full((height, img.shape[1], 3), 255, np.uint8)
    cv2.putText(b, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([b, img])


def render_row(det_name, clean_img, attacked_img, clean_dets, matched, lost, new):
    """Returns one figure row (clean | attacked) as an image."""
    left = cv2.resize(clean_img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(attacked_img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for c in clean_dets:
        draw_box(left, c["box"], f"{c['cls_name']} {c['conf']:.2f}", GREEN)
    for c, a, _ in matched:
        draw_box(right, c["box"], f"{c['cls_name']} {a['conf']:.2f}", GREEN)
    for c in lost:
        draw_box(right, c["box"], f"{c['cls_name']} 0.00", RED)
    for a in new:
        draw_box(right, a["box"], f"FP {a['cls_name']} {a['conf']:.2f}", ORANGE)
    left = banner(left, f"{det_name}  |  clean: {len(clean_dets)} detections")
    right = banner(right, f"attacked: {len(matched)} kept, {len(lost)} suppressed, {len(new)} false pos.")
    gap = np.full((left.shape[0], 6, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


def compose(rows):
    sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, sep, r])
    return out


def build_backends(spec, conf):
    """spec like 'ssd' or 'yolo:yolov8n.pt' -> (display_name, detect_fn)."""
    from visualize_forced_boxes import load_yolo, load_ssd      # same backends as the single-detector script
    name, _, arg = spec.partition(":")
    if name == "ssd":
        return "SSD300", load_ssd()
    if name == "yolo":
        w = arg or "yolov8n.pt"
        return Path(w).stem.replace("yolov", "YOLOv"), load_yolo(w)
    raise SystemExit(f"unknown detector spec {spec!r} (use ssd or yolo:<weights>)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--detectors", default="ssd,yolo:yolov8n.pt,yolo:yolov8m.pt")
    ap.add_argument("--out_dir", default="qualitative_figures")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--class_agnostic", action="store_true")
    ap.add_argument("--n_examples", type=int, default=3, help="frames per attack profile")
    ap.add_argument("--min_severity_pct", type=float, default=0.0,
                    help="only frames whose power x coverage is above this percentile (0-100) within their profile")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    from attack_effect_analysis import match_detections

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(config_path))
    root = config_path.parent.parent
    clean_base, final_dataset = root / cfg["paths"]["clean_base"], root / cfg["paths"]["final_dataset"]

    backends = [build_backends(s.strip(), args.conf) for s in args.detectors.split(",") if s.strip()]
    print("detectors:", [n for n, _ in backends])

    df = pd.read_csv(args.labels)
    if "pairing_verified" in df.columns:
        df = df[df["pairing_verified"] != False]
    if args.split:
        df = df[df["split"] == args.split]
    df = df[df["variation"] != "clean"].dropna(subset=["source_clean_image"])
    if args.min_severity_pct > 0 and {"power_mw", "coverage"} <= set(df.columns):
        df = df.assign(_sev=df["power_mw"] * df["coverage"])
        thr = df.groupby("variation")["_sev"].transform(lambda s: s.quantile(args.min_severity_pct / 100.0))
        df = df[df["_sev"] >= thr]

    index = []
    for var in df["variation"].unique():
        rows = df[df["variation"] == var].sample(frac=1.0, random_state=args.seed).to_dict("records")
        saved = 0
        for r in rows:
            if saved >= args.n_examples:
                break
            clean = cv2.imread(str(clean_base / r["source_clean_image"]))
            attacked = cv2.imread(str(final_dataset / str(r["path"]).replace("\\", "/")))
            if clean is None or attacked is None:
                continue
            clean = cv2.resize(clean, (224, 224))
            attacked = cv2.resize(attacked, (224, 224))

            fig_rows, summary, ok = [], {}, True
            for k, (name, detect) in enumerate(backends):
                cd = detect(clean, args.conf)
                if k == 0 and not cd:
                    ok = False            # first detector must see something in the clean frame
                    break
                ad = detect(attacked, args.conf)
                m, l, n = match_detections(cd, ad, args.iou_thresh, class_agnostic=args.class_agnostic)
                fig_rows.append(render_row(name, clean, attacked, cd, m, l, n))
                summary[name] = (len(cd), len(m), len(l), len(n))
            if not ok:
                continue
            stem = Path(r["source_clean_image"]).stem
            cv2.imwrite(str(out_dir / f"multi_{var}_{stem}.png"), compose(fig_rows))
            index.append({"variation": var, "source": r["source_clean_image"], "attacked_path": r["path"],
                          **{f"{k}_(clean,kept,suppressed,fp)": str(v) for k, v in summary.items()}})
            saved += 1
        print(f"[{var}] saved {saved}")

    pd.DataFrame(index).to_csv(out_dir / "qualitative_index.csv", index=False)
    print(f"\nfigures + qualitative_index.csv in {out_dir}/")
    print("Caption note: frames were drawn at random among attacked frames whose clean version has at "
          "least one detection for the first detector; illustrative, not a statistical sample.")


if __name__ == "__main__":
    main()
