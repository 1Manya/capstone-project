import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import sys

# =====================================================
# PATHS
# =====================================================

PROJECT_DIR = Path("/kaggle/working/capstone-project")
CODE_DIR = PROJECT_DIR / "Unet_ssd_fusion"

OUTPUT_DIR = PROJECT_DIR / "outputs/external_test"

UNET_FEATURES = OUTPUT_DIR / "external_unet_features.csv"
SSD_FEATURES = OUTPUT_DIR / "external_ssd_features.csv"

FUSION_CHECKPOINT = (
    PROJECT_DIR / "outputs/fusion/fusion_binary_final.pth"
)

METADATA_PATH = (
    PROJECT_DIR / "outputs/fusion/fusion_binary_metadata.json"
)

EXTERNAL_LABELS = Path(
    "/kaggle/input/datasets/manyai/nuscenes-attack-test/labels.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# LOAD ORIGINAL TRAINING METADATA
# =====================================================

with open(METADATA_PATH) as f:
    metadata = json.load(f)

feature_cols = metadata["feature_cols"]

scaler_mean = np.array(
    metadata["scaler_mean"], dtype=np.float32
)

scaler_scale = np.array(
    metadata["scaler_scale"], dtype=np.float32
)

assert len(feature_cols) == 10

# =====================================================
# LOAD EXTERNAL FEATURES
# =====================================================

unet_df = pd.read_csv(UNET_FEATURES)
ssd_df = pd.read_csv(SSD_FEATURES)
labels_df = pd.read_csv(EXTERNAL_LABELS)

print("U-Net features:", unet_df.shape)
print("SSD features:", ssd_df.shape)
print("External labels:", labels_df.shape)

# =====================================================
# MERGE FEATURES
# =====================================================

merged = unet_df.merge(
    ssd_df,
    on="path",
    how="inner",
    validate="one_to_one"
)

merged = merged.merge(
    labels_df,
    on="path",
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset:", merged.shape)

assert len(merged) == 2400, (
    f"Expected 2400 rows, got {len(merged)}"
)

assert merged["is_attacked"].notna().all()

# Check feature columns
missing = set(feature_cols) - set(merged.columns)

if missing:
    raise ValueError(f"Missing features: {missing}")

assert merged[feature_cols].notna().all().all()

print("\nAll 10 expected features found.")
print("Clean:", (merged["is_attacked"] == 0).sum())
print("Corrupt:", (merged["is_attacked"] == 1).sum())

# =====================================================
# APPLY ORIGINAL TRAINING SCALER
# DO NOT FIT A NEW SCALER
# =====================================================

X = merged[feature_cols].to_numpy(dtype=np.float32)

X_scaled = (X - scaler_mean) / scaler_scale

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
).to(DEVICE)

# =====================================================
# LOAD EXISTING FUSION MODEL
# INFERENCE ONLY
# =====================================================

sys.path.insert(0, str(CODE_DIR))

from fusion_model import FusionMLP

model = FusionMLP(
    input_dim=len(feature_cols),
    num_classes=2
).to(DEVICE)

state_dict = torch.load(
    FUSION_CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("\nExisting binary fusion checkpoint loaded.")
print("No training or fine-tuning performed.")

# =====================================================
# EXTERNAL INFERENCE
# =====================================================

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(
        probabilities, dim=1
    ).cpu().numpy()

probabilities = probabilities.cpu().numpy()

# Class mapping:
# 0 = Clean
# 1 = Corrupt

y_true = merged["is_attacked"].astype(int).to_numpy()

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_true, predictions)

precision = precision_score(
    y_true, predictions, zero_division=0
)

recall = recall_score(
    y_true, predictions, zero_division=0
)

f1 = f1_score(
    y_true, predictions, zero_division=0
)

auc = roc_auc_score(
    y_true, probabilities[:, 1]
)

cm = confusion_matrix(
    y_true, predictions, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n" + "=" * 60)
print("EXTERNAL TEST RESULTS — BINARY FUSION MODEL")
print("=" * 60)

print(f"Test samples       : {len(y_true)}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"Specificity        : {specificity:.4f}")
print(f"ROC-AUC            : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions,
        target_names=["Clean", "Corrupt"],
        digits=4
    )
)

# =====================================================
# SAVE PREDICTIONS
# =====================================================

results = merged.copy()

results["true_label"] = y_true
results["predicted_label"] = predictions

results["predicted_class"] = np.where(
    predictions == 0,
    "Clean",
    "Corrupt"
)

results["clean_probability"] = probabilities[:, 0]
results["corrupt_probability"] = probabilities[:, 1]

results["correct"] = (
    results["true_label"] == results["predicted_label"]
)

results_path = OUTPUT_DIR / "external_fusion_predictions.csv"

results.to_csv(results_path, index=False)

print("\nPredictions saved to:")
print(results_path)
