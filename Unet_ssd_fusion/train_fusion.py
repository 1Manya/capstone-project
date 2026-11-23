from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from config import UNET_DIR, SSD_DIR, FUSION_DIR, BATCH_SIZE, SEED
from common import load_labels, seed_everything
from fusion_model import FusionMLP

EPOCHS = 40

def main():
    seed_everything(SEED)
    FUSION_DIR.mkdir(parents=True, exist_ok=True)

    files = [UNET_DIR / "unet_features.csv", SSD_DIR / "ssd_features.csv"]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Complete U-Net and SSD feature extraction first."
            )

    labels = load_labels()
    merged = labels[["path", "split", "variation"]].copy()

    for p in files:
        feat = pd.read_csv(p)
        if "path" not in feat.columns:
            raise ValueError(f"{p} must contain a 'path' column.")
        merged = merged.merge(feat, on="path", how="inner", validate="one_to_one")

    feature_cols = [
        c for c in merged.columns
        if c.startswith(("unet_", "ssd_"))
    ]
    if not feature_cols:
        raise ValueError("No U-Net or SSD numeric feature columns found.")

    for c in feature_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged[feature_cols] = (
        merged[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    train = merged[merged["split"] == "train"].copy()
    val = merged[merged["split"] == "val"].copy()
    test = merged[merged["split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Expected non-empty train, val, and test splits in labels.csv.")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["variation"])
    y_val = encoder.transform(val["variation"])
    y_test = encoder.transform(test["variation"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols]).astype(np.float32)
    X_val = scaler.transform(val[feature_cols]).astype(np.float32)
    X_test = scaler.transform(test[feature_cols]).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP(X_train.shape[1], len(encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    ckpt_path = FUSION_DIR / "fusion_checkpoint.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["feature_cols"] != feature_cols or ckpt["classes"] != list(encoder.classes_):
            raise ValueError(
                "Existing fusion checkpoint schema differs. Move it aside to start fresh."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        print(f"Resuming fusion training from epoch {start_epoch + 1}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"Fusion epoch {epoch+1}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_pred = val_logits.argmax(1).cpu().numpy()

        val_acc = accuracy_score(y_val, val_pred)
        print(
            f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}, "
            f"val_accuracy={val_acc:.4f}"
        )

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "feature_cols": feature_cols,
            "classes": list(encoder.classes_),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, ckpt_path)

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_pred = test_logits.argmax(1).cpu().numpy()

    print("TEST ACCURACY:", accuracy_score(y_test, test_pred))
    print(classification_report(
        y_test, test_pred, target_names=encoder.classes_, zero_division=0
    ))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    torch.save(model.state_dict(), FUSION_DIR / "fusion_final.pth")
    with open(FUSION_DIR / "fusion_metadata.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "classes": list(encoder.classes_),
            "inputs": ["U-Net segmentation features", "SSD detection features"]
        }, f, indent=2)

if __name__ == "__main__":
    main()
