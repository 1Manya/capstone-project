from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import IMAGE_SIZE, MASK_ROOT, UNET_DIR, OUTPUT_ROOT
from common import load_labels, resolve_image_path, seed_everything
from unet_model import UNet

OUT = UNET_DIR / "unet_features.csv"
BATCH_SAVE = 100

def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    weights = UNET_DIR / "unet_final.pth"
    if not weights.exists():
        weights = UNET_DIR / "checkpoint_last.pth"
        if not weights.exists():
            raise FileNotFoundError("Train U-Net first; checkpoint not found.")
        state = torch.load(weights, map_location=device)["model"]
    else:
        state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    df = load_labels()
    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = set(old["path"].astype(str))
    pending = df[~df["path"].astype(str).isin(done)].copy()
    rows = []
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="U-Net feature extraction"):
        p = resolve_image_path(row["path"], row["variation"])
        image = cv2.imread(str(p))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        x = torch.from_numpy(image.transpose(2,0,1).copy()).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
        binary = (prob >= 0.5).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
        rows.append({
            "path": str(row["path"]),
            "unet_mask_coverage": float(binary.mean()),
            "unet_mean_probability": float(prob.mean()),
            "unet_max_probability": float(prob.max()),
            "unet_component_count": int((areas >= 5).sum()),
            "unet_largest_component": int(areas.max()) if len(areas) else 0,
        })
        if len(rows) >= BATCH_SAVE:
            append_rows(rows); rows = []
    if rows:
        append_rows(rows)
    print(f"Features saved to {OUT}")

def append_rows(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)

if __name__ == "__main__":
    main()
