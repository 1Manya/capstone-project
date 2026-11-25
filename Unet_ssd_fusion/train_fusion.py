```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from tqdm.auto import tqdm

from config import UNET_DIR, SSD_DIR, FUSION_DIR, BATCH_SIZE, SEED
from common import load_labels, seed_everything
from fusion_model import FusionMLP


EPOCHS = 40
CLASS_NAMES = ["clean", "corrupt"]


def main():

    seed_everything(SEED)
    FUSION_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load U-Net and SSD features
    # --------------------------------------------------

    files = [
        UNET_DIR / "unet_features.csv",
        SSD_DIR / "ssd_features.csv"
    ]

    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"Missing feature file: {p}")

    labels = load_labels()

    # Use label instead of variation
    merged = labels[["path", "split", "label"]].copy()

    merged["path"] = (
        merged["path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    # --------------------------------------------------
    # 2. Merge features
    # --------------------------------------------------

    for p in files:

        feat = pd.read_csv(p)

        if "path" not in feat.columns:
            raise ValueError(f"{p} must contain a path column.")

        feat["path"] = (
            feat["path"]
            .astype(str)
            .str.replace("\\", "/", regex=False)
        )

        # Remove metadata to prevent target leakage
        metadata = [
            "variation",
            "split",
            "source_clean_image",
            "is_attacked"
        ]

        feat = feat.drop(
            columns=[c for c in metadata if c in feat.columns]
        )

        # Prefix SSD features
        if "ssd" in p.name.lower():
            feat = feat.rename(columns={
                c: f"ssd_{c}"
                for c in feat.columns
                if c != "path"
            })

        merged = merged.merge(
            feat,
            on="path",
            how="inner",
            validate="one_to_one"
        )

    # --------------------------------------------------
    # 3. Create binary target
    # --------------------------------------------------

    merged["target"] = (
        merged["label"].astype(str).str.lower() != "clean"
    ).astype(int)

    feature_cols = [
        c for c in merged.columns
        if c.startswith(("unet_", "ssd_"))
    ]

    if not feature_cols:
        raise ValueError("No U-Net or SSD feature columns found.")

    for c in feature_cols:
        merged[c] = pd.to_numeric(
            merged[c], errors="coerce"
        )

    merged[feature_cols] = (
        merged[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    print("\nMerged dataset shape:", merged.shape)
    print("\nBinary class distribution:")
    print(merged["target"].value_counts().rename({
        0: "clean",
        1: "corrupt"
    }))

    # --------------------------------------------------
    # 4. Train / validation / test split
    # --------------------------------------------------

    train = merged[merged["split"] == "train"].copy()
    val = merged[merged["split"] == "val"].copy()
    test = merged[merged["split"] == "test"].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, or test split is empty.")

    y_train = train["target"].to_numpy(dtype=np.int64)
    y_val = val["target"].to_numpy(dtype=np.int64)
    y_test = test["target"].to_numpy(dtype=np.int64)

    # Fit scaler only on training data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[feature_cols]
    ).astype(np.float32)

    X_val = scaler.transform(
        val[feature_cols]
    ).astype(np.float32)

    X_test = scaler.transform(
        test[feature_cols]
    ).astype(np.float32)

    print("\nFeature count:", len(feature_cols))
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # --------------------------------------------------
    # 5. Model and class-weighted loss
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nTraining device:", device)

    model = FusionMLP(
        input_dim=X_train.shape[1],
        num_classes=2
    ).to(device)

    class_counts = np.bincount(y_train, minlength=2)

    class_weights = (
        len(y_train) / (2.0 * class_counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    print("\nClass counts:", class_counts)
    print("Class weights:", class_weights.cpu().numpy())

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # Separate checkpoint from the old six-class model
    ckpt_path = FUSION_DIR / "fusion_binary_checkpoint.pth"

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    # --------------------------------------------------
    # 6. Training
    # --------------------------------------------------

    best_val_f1 = -1.0
    best_model_state = None

    for epoch in range(EPOCHS):

        model.train()
        losses = []

        for xb, yb in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        ):

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Validation
        model.eval()

        with torch.no_grad():

            val_logits = model(
                torch.from_numpy(X_val).to(device)
            )

            val_pred = val_logits.argmax(dim=1).cpu().numpy()

        val_acc = accuracy_score(y_val, val_pred)

        val_f1 = f1_score(
            y_val,
            val_pred,
            average="binary",
            zero_division=0
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {np.mean(losses):.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Save best model based on validation F1
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "feature_cols": feature_cols,
                "classes": CLASS_NAMES,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist()
            }, ckpt_path)

            print("Best binary checkpoint saved.")

    # --------------------------------------------------
    # 7. Load best model
    # --------------------------------------------------

    if best_model_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    model.load_state_dict(best_model_state)
    model.eval()

    # --------------------------------------------------
    # 8. Test evaluation
    # --------------------------------------------------

    with torch.no_grad():

        test_logits = model(
            torch.from_numpy(X_test).to(device)
        )

        test_probabilities = torch.softmax(
            test_logits, dim=1
        ).cpu().numpy()

        test_predictions = test_probabilities.argmax(axis=1)

    print("\n========== TEST RESULTS ==========")

    print(
        "Test Accuracy:",
        accuracy_score(y_test, test_predictions)
    )

    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            test_predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:\n",
        confusion_matrix(y_test, test_predictions)
    )

    # --------------------------------------------------
    # 9. Save final model and metadata
    # --------------------------------------------------

    final_path = FUSION_DIR / "fusion_binary_final.pth"

    torch.save(model.state_dict(), final_path)

    with open(
        FUSION_DIR / "fusion_binary_metadata.json",
        "w"
    ) as f:

        json.dump({
            "task": "binary_clean_corrupt_classification",
            "classes": CLASS_NAMES,
            "feature_cols": feature_cols,
            "inputs": [
                "U-Net segmentation features",
                "SSD detection features"
            ],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "best_validation_f1": float(best_val_f1)
        }, f, indent=2)

    # --------------------------------------------------
    # 10. Save test predictions and probabilities
    # --------------------------------------------------

    results = test[["path", "label"]].copy()

    results["true_class"] = np.where(
        y_test == 0, "clean", "corrupt"
    )

    results["predicted_class"] = np.where(
        test_predictions == 0, "clean", "corrupt"
    )

    results["clean_probability"] = test_probabilities[:, 0]
    results["corrupt_probability"] = test_probabilities[:, 1]

    results.to_csv(
        FUSION_DIR / "binary_test_predictions.csv",
        index=False
    )

    print("\nSaved:")
    print(final_path)
    print(ckpt_path)
    print(FUSION_DIR / "fusion_binary_metadata.json")
    print(FUSION_DIR / "binary_test_predictions.csv")


if __name__ == "__main__":
    main()
```
