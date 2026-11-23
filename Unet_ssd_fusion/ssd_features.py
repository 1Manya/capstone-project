from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from tqdm.auto import tqdm

from config import IMAGE_SIZE, SSD_DIR
from common import load_labels, resolve_image_path, seed_everything

OUT = SSD_DIR / "ssd_features.csv"
CONF_THRESHOLD = 0.25
SAVE_EVERY = 100

def extract_one(model, image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (300, 300))
    tensor = torch.from_numpy(rgb.transpose(2,0,1).copy()).float().to(device) / 255.0
    with torch.no_grad():
        pred = model([tensor])[0]
    scores = pred["scores"].detach().cpu().numpy()
    boxes = pred["boxes"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    keep = scores >= CONF_THRESHOLD
    scores, boxes, labels = scores[keep], boxes[keep], labels[keep]
    if len(scores) == 0:
        return {
            "ssd_num_detections": 0, "ssd_mean_confidence": 0.0,
            "ssd_max_confidence": 0.0, "ssd_mean_box_area": 0.0,
            "ssd_mean_box_aspect": 0.0, "ssd_mean_box_center_y": 0.0,
            "ssd_class_count": 0
        }
    widths = np.maximum(0, boxes[:,2] - boxes[:,0])
    heights = np.maximum(0, boxes[:,3] - boxes[:,1])
    areas = widths * heights / float(300 * 300)
    aspects = widths / np.maximum(heights, 1e-6)
    centers_y = (boxes[:,1] + boxes[:,3]) / 600.0
    return {
        "ssd_num_detections": int(len(scores)),
        "ssd_mean_confidence": float(scores.mean()),
        "ssd_max_confidence": float(scores.max()),
        "ssd_mean_box_area": float(areas.mean()),
        "ssd_mean_box_aspect": float(aspects.mean()),
        "ssd_mean_box_center_y": float(centers_y.mean()),
        "ssd_class_count": int(len(np.unique(labels)))
    }

def append_rows(rows):
    SSD_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)

def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SSD_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading pretrained SSD300 VGG16 COCO weights...")
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(device).eval()

    df = load_labels()
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["path"])["path"].astype(str))
    pending = df[~df["path"].astype(str).isin(done)]
    rows = []
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="SSD feature extraction"):
        try:
            img_path = resolve_image_path(row["path"], row["variation"])
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            feats = extract_one(model, image, device)
            feats["path"] = str(row["path"])
            rows.append(feats)
        except Exception as e:
            print(f"Skipped {row['path']}: {e}")
        if len(rows) >= SAVE_EVERY:
            append_rows(rows); rows = []
    if rows:
        append_rows(rows)
    print(f"SSD features saved to {OUT}")

if __name__ == "__main__":
    main()
